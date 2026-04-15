"""
emp_policy.py — SE(3) EMP: Training, Adaptation, Rollout
=========================================================
All EMP mathematics lives here.  Zero coupling to any scene or task.

The only external dependencies are:
  • numpy / scipy / sklearn / cvxpy  (maths)
  • pb_utils  (quaternion primitives)
  • pb_stage1_env  (fk, ik, PandaSimEnv — only for PyBullet execution helper)

To switch tasks, change the scene passed to execute_policy_in_sim().
The EMP model dict has no scene-specific fields.

Public API
----------
preprocess_demo(demo)                → data dict
train_emp(data)                      → model dict
adapt_emp(model, new_x_start, ...)   → adapted model dict
rollout_se3(x0, q0, model, ...)      → (pos_traj, quat_traj)
execute_policy_in_sim(model, traj_pos, traj_quat, scene, sim)
stability_check(Ak_list, P)          → (n_stable, n_total)
lyapunov_violation(X, Xdot, P, x*)  → float ∈ [0,1]
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as Rot
from sklearn.mixture import GaussianMixture
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from pb_utils import (
    quat_log as _quat_log_raw,
    quat_exp,
    quat_conjugate,
    quat_mul,
)
from pb_stage1_env import (
    fk, ik, Q_NEUTRAL,
    make_pose, pose_to_quat,
    _get_sim, PandaSimEnv,
)

np.random.seed(0)

# Colour palette (used in plots)
C_DEMO   = "#3B8BD4"
C_ORIG   = "#1D9E75"
C_ADAPT  = "#D85A30"
C_ADAPT2 = "#9933CC"
C_ADAPT3 = "#CC3366"
C_JOINT  = "#BA7517"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_demo(demo: dict, N_max: int = 150) -> dict:
    """
    Extract position + quaternion training arrays from one demonstration dict.

    Parameters
    ----------
    demo  : dict from TaskSpec.generate_demo (or load_demos)
    N_max : downsample to at most this many timesteps

    Returns
    -------
    dict with X, Xdot, Q, Qdot, x_star, q_star, dt
    """
    pos  = demo['pos']
    quat = demo['quat']
    t    = demo['times']
    dt   = float(np.diff(t).mean()) if len(t) > 1 else 0.01

    idx  = np.round(np.linspace(0, len(pos) - 1,
                                min(N_max, len(pos)))).astype(int)
    pos  = pos[idx]
    quat = quat[idx]

    quat = normalise_quat_sequence(quat)
    Xdot = np.gradient(pos,  axis=0) / dt
    Qdot = np.zeros((len(quat), 3))
    for i in range(1, len(quat)):
        dq = quat_mul(quat[i], quat_conjugate(quat[i - 1]))
        if dq[0] < 0.0:
            dq *= -1.0
        Qdot[i] = _quat_log_raw(dq) / dt
    if len(Qdot) > 1:
        Qdot[0] = Qdot[1]

    x_star = pos[-1].copy()
    q_star = quat[-1].copy()

    return dict(
        X=pos[:-1], Xdot=Xdot[:-1],
        Q=quat[:-1], Qdot=Qdot[:-1],
        x_star=x_star, q_star=q_star, dt=dt,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Position GMM
# ══════════════════════════════════════════════════════════════════════════════

def fit_pos_gmm(X: np.ndarray, k_min: int = 2, k_max: int = 6,
                n_init: int = 5) -> GaussianMixture:
    best, bic = None, np.inf
    for k in range(k_min, k_max + 1):
        g = GaussianMixture(k, covariance_type='full',
                            n_init=n_init, random_state=0)
        g.fit(X)
        b = g.bic(X)
        if b < bic:
            bic, best = b, g
    return best


def sort_gmm(gmm: GaussianMixture,
             x_start: np.ndarray, x_end: np.ndarray) -> np.ndarray:
    d = x_end - x_start
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.arange(gmm.n_components)
    return np.argsort(gmm.means_ @ (d / n))


def gamma_pos_batch(X: np.ndarray, means: np.ndarray,
                    covs: np.ndarray, priors: np.ndarray) -> np.ndarray:
    N, K = len(X), len(means)
    lp = np.zeros((N, K))
    for k in range(K):
        diff = X - means[k]
        ci   = np.linalg.inv(covs[k])
        lds  = np.linalg.slogdet(covs[k])[1]
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:, k] = (np.log(priors[k] + 1e-300)
                    - 0.5 * (3 * np.log(2 * np.pi) + lds + maha))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / (p.sum(1, keepdims=True) + 1e-300)


def gamma_pos_single(x: np.ndarray, means: np.ndarray,
                     covs: np.ndarray, priors: np.ndarray) -> np.ndarray:
    return gamma_pos_batch(x[None], means, covs, priors)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Orientation (Quaternion) GMM on tangent space
# ══════════════════════════════════════════════════════════════════════════════

def log_quat(q_wxyz: np.ndarray, q_att_wxyz: np.ndarray) -> np.ndarray:
    """Riemannian log map: project q onto tangent space at q_att."""
    qa  = q_att_wxyz / np.linalg.norm(q_att_wxyz)
    qn  = q_wxyz / np.linalg.norm(q_wxyz)
    if np.dot(qn, qa) < 0.0:
        qn = -qn
    qc  = quat_conjugate(qa)
    q_rel = quat_mul(qc, qn)
    w   = np.clip(q_rel[0], -1.0, 1.0)
    v   = q_rel[1:]
    nv  = np.linalg.norm(v)
    if nv < 1e-10:
        return np.zeros(3)
    return (np.arccos(w) / nv) * v


def normalise_quat_sequence(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float).copy()
    if len(Q) == 0:
        return Q
    Q[0] /= np.linalg.norm(Q[0])
    for i in range(1, len(Q)):
        if np.dot(Q[i - 1], Q[i]) < 0.0:
            Q[i] *= -1.0
        Q[i] /= np.linalg.norm(Q[i])
    return Q


def exp_quat(omega: np.ndarray, q_att_wxyz: np.ndarray) -> np.ndarray:
    """Riemannian exp map: lift omega ∈ ℝ³ back to unit quaternion."""
    qa  = q_att_wxyz / np.linalg.norm(q_att_wxyz)
    q_d = quat_exp(omega)
    return quat_mul(qa, q_d)


def project_quats_to_tangent(Q: np.ndarray,
                              q_att: np.ndarray) -> np.ndarray:
    return np.array([log_quat(Q[i], q_att) for i in range(len(Q))])


def fit_ori_gmm(Q: np.ndarray, q_star: np.ndarray,
                K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q_tan = project_quats_to_tangent(Q, q_star)
    g = GaussianMixture(K, covariance_type='full', n_init=5, random_state=0)
    g.fit(Q_tan)
    return g.means_, g.covariances_, g.weights_


def gamma_ori_batch(Q: np.ndarray, mu3d: np.ndarray,
                    cov3d: np.ndarray, priors: np.ndarray,
                    q_star: np.ndarray) -> np.ndarray:
    Q_tan = project_quats_to_tangent(Q, q_star)
    N, K  = len(Q_tan), len(mu3d)
    lp    = np.zeros((N, K))
    for k in range(K):
        diff = Q_tan - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6 * np.eye(3))
        lds  = np.linalg.slogdet(cov3d[k] + 1e-6 * np.eye(3))[1]
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:, k] = (np.log(priors[k] + 1e-300)
                    - 0.5 * (3 * np.log(2 * np.pi) + lds + maha))
    lp -= lp.max(1, keepdims=True)
    p  = np.exp(lp)
    return p / (p.sum(1, keepdims=True) + 1e-300)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Joint chain & local-frame storage (for Laplacian adaptation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_joints(means: np.ndarray, covs: np.ndarray,
                   x_start: np.ndarray, x_end: np.ndarray) -> np.ndarray:
    K = len(means)
    interior = []
    for k in range(K - 1):
        Si  = np.linalg.inv(covs[k]     + 1e-8 * np.eye(3))
        Si1 = np.linalg.inv(covs[k + 1] + 1e-8 * np.eye(3))
        St  = np.linalg.inv(Si + Si1)
        interior.append(St @ (Si @ means[k] + Si1 @ means[k + 1]))
    rows = ([x_start[None]]
            + ([np.array(interior)] if interior else [])
            + [x_end[None]])
    return np.vstack(rows)


def local_frame(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    e = b - a
    d = np.linalg.norm(e)
    if d < 1e-12:
        return np.eye(3), 0.0
    e /= d
    t = np.array([1., 0., 0.]) if abs(e[0]) < 0.9 else np.array([0., 1., 0.])
    t -= e * (e @ t)
    t /= np.linalg.norm(t)
    return np.column_stack([e, t, np.cross(e, t)]), d


def store_frames(means: np.ndarray, covs: np.ndarray,
                 joints: np.ndarray) -> List[dict]:
    frames = []
    for k in range(len(means)):
        R, d   = local_frame(joints[k], joints[k + 1])
        mu_loc = R.T @ (means[k] - joints[k])
        ev, E  = np.linalg.eigh(covs[k])
        E_loc  = R.T @ E
        frames.append(dict(mu_loc=mu_loc, E_loc=E_loc, eigvals=ev, dist=d))
    return frames


def recover_gmm_params(joints_new: np.ndarray, frames: List[dict],
                       priors: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(frames)
    means_n, covs_n = [], []
    for k in range(K):
        R_new, d_new = local_frame(joints_new[k], joints_new[k + 1])
        s   = d_new / frames[k]['dist'] if frames[k]['dist'] > 1e-12 else 1.0
        mu  = joints_new[k] + R_new @ frames[k]['mu_loc']
        E   = R_new @ frames[k]['E_loc']
        ev  = np.maximum(frames[k]['eigvals'] * s, 1e-8)
        cov = E @ np.diag(ev) @ E.T
        means_n.append(mu)
        covs_n.append(0.5 * (cov + cov.T))
    return np.array(means_n), np.array(covs_n), priors.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Laplacian editing
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(n: int) -> np.ndarray:
    L = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if i == 0:
            L[0, 0] = 1.;  L[0, 1] = -1.
        elif i == n:
            L[n, n] = 1.;  L[n, n - 1] = -1.
        else:
            L[i, i] = 1.;  L[i, i - 1] = L[i, i + 1] = -0.5
    return L


def laplacian_edit(joints: np.ndarray, L: np.ndarray,
                   Delta: np.ndarray,
                   new_start: np.ndarray, new_end: np.ndarray) -> np.ndarray:
    n = len(joints) - 1
    if n == 1:
        return np.vstack([new_start, new_end])
    L_int = L[1:n, 1:n]
    rhs   = (Delta[1:n]
             - np.outer(L[1:n, 0], new_start)
             - np.outer(L[1:n, n], new_end))
    beta  = np.linalg.solve(L_int, rhs)
    return np.vstack([new_start, beta, new_end])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Velocity profile regeneration
# ══════════════════════════════════════════════════════════════════════════════

def create_velocity_profile(joints: np.ndarray, N_pts: int = 150,
                            dt: float = 0.01
                            ) -> Tuple[np.ndarray, np.ndarray]:
    segs  = np.diff(joints, axis=0)
    dists = np.linalg.norm(segs, axis=1)
    total = dists.sum()
    if total < 1e-12:
        return (np.tile(joints[0], (N_pts, 1)),
                np.zeros((N_pts, joints.shape[1])))
    cum  = np.concatenate([[0.], np.cumsum(dists)])
    lam  = cum / total
    idx_j = np.clip(np.round(lam * (N_pts - 1)).astype(int), 0, N_pts - 1)
    t_u   = np.linspace(0, 1, N_pts)
    pos   = np.column_stack(
        [np.interp(t_u, lam, joints[:, d]) for d in range(joints.shape[1])]
    )
    L      = build_laplacian(N_pts - 1)
    Delta  = L @ pos
    A_a    = np.vstack([L, np.eye(N_pts)[idx_j]])
    b_a    = np.vstack([Delta, joints])
    pos    = np.linalg.lstsq(A_a.T @ A_a, A_a.T @ b_a, rcond=None)[0]
    vel    = np.gradient(pos, axis=0) / dt
    return pos, vel


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Lyapunov P estimation
# ══════════════════════════════════════════════════════════════════════════════

def gmm_centroids(X: np.ndarray, Xdot: np.ndarray,
                  means: np.ndarray, covs: np.ndarray,
                  priors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g      = gamma_pos_batch(X, means, covs, priors)
    assign = np.argmax(g, axis=1)
    K = len(means)
    Xb, Xdb = [], []
    for k in range(K):
        m = assign == k
        Xb.append(X[m].mean(0)     if m.sum() > 0 else means[k])
        Xdb.append(Xdot[m].mean(0) if m.sum() > 0 else np.zeros(3))
    return np.array(Xb), np.array(Xdb)


def estimate_P(X_bar: np.ndarray, Xdot_bar: np.ndarray,
               x_star: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    d, K = 3, len(X_bar)
    Z    = X_bar - x_star
    P_v  = cp.Variable((d, d), symmetric=True)
    s    = cp.Variable(K, nonneg=True)
    cons = [P_v >> float(eps) * np.eye(d)]
    for k in range(K):
        cons.append(s[k] >= Xdot_bar[k] @ P_v @ Z[k])
    cp.Problem(cp.Minimize(cp.sum(s)), cons).solve(
        solver=cp.SCS, verbose=False, eps=1e-5)
    Pv = P_v.value
    if Pv is None or not np.all(np.isfinite(Pv)):
        Pv = np.eye(d)
    Pv = 0.5 * (Pv + Pv.T)
    ev = float(np.linalg.eigvalsh(Pv).min())
    ep = float(np.asarray(eps).flat[0])
    if ev < ep:
        Pv += (ep - ev + 1e-8) * np.eye(d)
    return Pv


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Stability-constrained SDP for {A_k}
# ══════════════════════════════════════════════════════════════════════════════

def project_stable_pos(Ak: np.ndarray, P: np.ndarray,
                       eps: float = 1e-3) -> np.ndarray:
    M   = Ak.T @ P + P @ Ak
    lam = float(np.linalg.eigvalsh(M).max())
    if lam >= -eps:
        alpha = (lam + 2 * eps) / (2 * float(np.linalg.eigvalsh(P).min()))
        Ak    = Ak - alpha * np.eye(Ak.shape[0])
    return Ak


def project_stable_ori(Ak: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    lam = float(np.linalg.eigvalsh(Ak + Ak.T).max())
    if lam >= -eps:
        Ak = Ak - ((lam + 2 * eps) / 2) * np.eye(Ak.shape[0])
    return Ak


def estimate_Ak_pos(X: np.ndarray, Xdot: np.ndarray,
                    means: np.ndarray, covs: np.ndarray, priors: np.ndarray,
                    P: np.ndarray, x_star: np.ndarray,
                    eps: float = 1e-3) -> List[np.ndarray]:
    K_l = len(means);  d = 3
    Z   = X - x_star
    g   = gamma_pos_batch(X, means, covs, priors)
    Av  = [cp.Variable((d, d)) for _ in range(K_l)]
    pred = sum(cp.multiply(g[:, k:k+1], Z @ Av[k].T) for k in range(K_l))
    obj  = cp.Minimize(cp.sum_squares(Xdot - pred) / len(X))
    cons = [Av[k].T @ P + P @ Av[k] << -float(eps) * np.eye(d)
            for k in range(K_l)]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False,
                                eps=1e-4, max_iters=15000)
    result = []
    for k in range(K_l):
        Ak = Av[k].value
        if Ak is None or not np.all(np.isfinite(Ak)):
            Ak = -np.eye(d)
        result.append(project_stable_pos(Ak, P, eps=float(eps)))
    return result


def estimate_Ak_ori(Q_tan: np.ndarray, Qdot_tan: np.ndarray,
                    mu3d: np.ndarray, cov3d: np.ndarray, priors: np.ndarray,
                    eps: float = 1e-3) -> List[np.ndarray]:
    K_l = len(mu3d);  d = 3;  N = len(Q_tan)
    lp  = np.zeros((N, K_l))
    for k in range(K_l):
        diff = Q_tan - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6 * np.eye(3))
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:, k] = np.log(priors[k] + 1e-300) - 0.5 * maha
    lp -= lp.max(1, keepdims=True)
    g   = np.exp(lp)
    g  /= g.sum(1, keepdims=True) + 1e-300

    Av   = [cp.Variable((d, d)) for _ in range(K_l)]
    pred = sum(cp.multiply(g[:, k:k+1], Q_tan @ Av[k].T) for k in range(K_l))
    obj  = cp.Minimize(cp.sum_squares(Qdot_tan - pred) / N)
    cons = [Av[k] + Av[k].T << -float(eps) * np.eye(d) for k in range(K_l)]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False,
                                eps=1e-4, max_iters=10000)
    result = []
    for k in range(K_l):
        Ak = Av[k].value
        if Ak is None or not np.all(np.isfinite(Ak)):
            Ak = -np.eye(d)
        result.append(project_stable_ori(Ak, eps=float(eps)))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Runtime velocity computation
# ══════════════════════════════════════════════════════════════════════════════

def pos_velocity(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
                 priors: np.ndarray, Ak_pos: List[np.ndarray],
                 x_star: np.ndarray, g_ood: float = 1.5) -> np.ndarray:
    gamma = gamma_pos_single(x, means, covs, priors)
    z     = x - x_star
    v     = sum(gamma[k] * (Ak_pos[k] @ z) for k in range(len(Ak_pos)))
    g     = 0.25 + g_ood * (1.0 - float(gamma.max()))
    return v - g * z


def ori_velocity(q: np.ndarray, mu3d: np.ndarray, cov3d: np.ndarray,
                 priors: np.ndarray, Ak_ori: List[np.ndarray],
                 q_star: np.ndarray) -> np.ndarray:
    K     = len(mu3d)
    log_q = log_quat(q, q_star)
    g     = np.zeros(K)
    for k in range(K):
        diff = log_q - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6 * np.eye(3))
        g[k] = np.exp(-0.5 * (diff @ ci @ diff) + np.log(priors[k] + 1e-300))
    g /= g.sum() + 1e-300
    return sum(g[k] * (Ak_ori[k] @ log_q) for k in range(K))


# ══════════════════════════════════════════════════════════════════════════════
# 9b. Obstacle avoidance modulation
# ══════════════════════════════════════════════════════════════════════════════

def modulate_velocity_obstacle(
        x: np.ndarray,
        xdot: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = 0.03,
        influence_radius: float = 0.12,
        gain: float = 0.5,
) -> np.ndarray:
    """
    Modulate the nominal EMP velocity to avoid a spherical obstacle.

    Uses the modulation approach from the EMP paper: when the EE enters the
    influence radius of an obstacle, a repulsive component is added that
    pushes the trajectory away from the obstacle surface while preserving
    convergence to the goal.

    Parameters
    ----------
    x                : current EE position (3,)
    xdot             : nominal EMP velocity (3,)
    obstacle_pos     : obstacle centre (3,)
    obstacle_radius  : obstacle radius (m)
    influence_radius : radius of influence zone around obstacle (m)
    gain             : strength of repulsion

    Returns
    -------
    modulated velocity (3,)
    """
    diff = x - obstacle_pos
    dist = np.linalg.norm(diff)
    safety_margin = obstacle_radius + 0.01  # extra margin

    if dist > influence_radius or dist < 1e-9:
        return xdot

    # Normal direction from obstacle to EE
    n = diff / dist

    # How deep into the influence zone (0 = at boundary, 1 = at surface)
    alpha = 1.0 - (dist - safety_margin) / (influence_radius - safety_margin)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Project velocity onto tangent plane of obstacle surface
    v_normal = np.dot(xdot, n) * n
    v_tangent = xdot - v_normal

    # If moving toward obstacle, add repulsion
    if v_normal.dot(n) < 0:
        repulsion = -gain * alpha * v_normal
        xdot_mod = v_tangent + repulsion
    else:
        xdot_mod = xdot

    # Scale to preserve nominal speed magnitude
    nominal_speed = np.linalg.norm(xdot)
    if nominal_speed > 1e-9:
        mod_speed = np.linalg.norm(xdot_mod)
        if mod_speed > 1e-9:
            xdot_mod = xdot_mod * (nominal_speed / mod_speed)

    return xdot_mod


def rollout_se3_with_obstacles(
        x0: np.ndarray,
        q0: np.ndarray,
        model: dict,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = 0.03,
        influence_radius: float = 0.12,
        obstacle_gain: float = 0.5,
        dt: float = None,
        max_steps: int = 600,
        pos_tol: float = 0.01,
        ori_tol: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SE(3) EMP rollout with spherical obstacle avoidance modulation.

    Parameters
    ----------
    x0, q0           : initial EE position (3,) and quaternion wxyz (4,)
    model            : dict from train_emp or adapt_emp
    obstacle_pos     : obstacle centre (3,)
    obstacle_radius  : obstacle radius (m)
    influence_radius : radius of influence zone (m)
    obstacle_gain    : repulsion strength
    dt, max_steps, pos_tol, ori_tol : same as rollout_se3

    Returns
    -------
    pos_traj  : (T, 3)
    quat_traj : (T, 4) wxyz
    """
    if dt is None:
        dt = model['dt']

    x = x0.copy()
    q = q0.copy()
    q /= np.linalg.norm(q)

    pos_t  = [x.copy()]
    quat_t = [q.copy()]

    means_p  = model['means']
    covs_p   = model['covs']
    priors_p = model['priors']
    Ak_pos   = model['Ak_pos']
    x_star   = model['x_star']
    mu3d     = model['mu3d']
    cov3d    = model['cov3d']
    priors_o = model['priors_o']
    Ak_ori   = model['Ak_ori']
    q_star   = model['q_star']

    prev_err = np.linalg.norm(x - x_star)
    stall_count = 0
    for _ in range(max_steps):
        xdot = pos_velocity(x, means_p, covs_p, priors_p, Ak_pos, x_star)

        # Apply obstacle modulation
        xdot = modulate_velocity_obstacle(
            x, xdot, obstacle_pos, obstacle_radius,
            influence_radius, obstacle_gain,
        )

        spd = np.linalg.norm(xdot)
        if spd > 1e-9:
            xdot = xdot * min(1.0, 0.25 / spd)
        x = x + dt * xdot

        omega = ori_velocity(q, mu3d, cov3d, priors_o, Ak_ori, q_star)
        omg = np.linalg.norm(omega)
        if omg > 1e-9:
            omega = omega * min(1.0, 1.5 / omg)
        dq    = quat_exp(dt * omega)
        q     = quat_mul(dq, q)
        q    /= np.linalg.norm(q)

        pos_t.append(x.copy())
        quat_t.append(q.copy())

        pos_err = np.linalg.norm(x - x_star)
        ori_err = np.linalg.norm(log_quat(q, q_star))
        if pos_err < pos_tol and ori_err < ori_tol:
            break
        if prev_err - pos_err < 5e-5:
            stall_count += 1
        else:
            stall_count = 0
        prev_err = pos_err
        if stall_count >= 40 and pos_err < 0.08:
            break

    pos_err = np.linalg.norm(pos_t[-1] - x_star)
    ori_err = np.linalg.norm(log_quat(quat_t[-1], q_star))
    if pos_err < 0.08 and (pos_err > pos_tol or ori_err > ori_tol):
        x_cur = pos_t[-1].copy()
        q_cur = quat_t[-1].copy()
        for alpha in np.linspace(0.2, 1.0, 5):
            x_blend = (1.0 - alpha) * x_cur + alpha * x_star
            q_blend = exp_quat(alpha * log_quat(q_star, q_cur), q_cur)
            pos_t.append(x_blend.copy())
            quat_t.append(q_blend / np.linalg.norm(q_blend))

    return np.array(pos_t), np.array(quat_t)


# ══════════════════════════════════════════════════════════════════════════════
# 9b. Obstacle avoidance modulation
# ══════════════════════════════════════════════════════════════════════════════

def modulate_velocity_obstacle(
        x: np.ndarray,
        xdot: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = 0.03,
        influence_radius: float = 0.12,
        gain: float = 0.5,
) -> np.ndarray:
    """
    Modulate the nominal EMP velocity to avoid a spherical obstacle.

    Uses the modulation approach from the EMP paper: when the EE enters the
    influence radius of an obstacle, a repulsive component is added that
    pushes the trajectory away from the obstacle surface while preserving
    convergence to the goal.

    Parameters
    ----------
    x                : current EE position (3,)
    xdot             : nominal EMP velocity (3,)
    obstacle_pos     : obstacle centre (3,)
    obstacle_radius  : obstacle radius (m)
    influence_radius : radius of influence zone around obstacle (m)
    gain             : strength of repulsion

    Returns
    -------
    modulated velocity (3,)
    """
    diff = x - obstacle_pos
    dist = np.linalg.norm(diff)
    safety_margin = obstacle_radius + 0.01  # extra margin

    if dist > influence_radius or dist < 1e-9:
        return xdot

    # Normal direction from obstacle to EE
    n = diff / dist

    # How deep into the influence zone (0 = at boundary, 1 = at surface)
    alpha = 1.0 - (dist - safety_margin) / (influence_radius - safety_margin)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Project velocity onto tangent plane of obstacle surface
    v_normal = np.dot(xdot, n) * n
    v_tangent = xdot - v_normal

    # If moving toward obstacle, add repulsion
    if v_normal.dot(n) < 0:
        repulsion = -gain * alpha * v_normal
        xdot_mod = v_tangent + repulsion
    else:
        xdot_mod = xdot

    # Scale to preserve nominal speed magnitude
    nominal_speed = np.linalg.norm(xdot)
    if nominal_speed > 1e-9:
        mod_speed = np.linalg.norm(xdot_mod)
        if mod_speed > 1e-9:
            xdot_mod = xdot_mod * (nominal_speed / mod_speed)

    return xdot_mod


def rollout_se3_with_obstacles(
        x0: np.ndarray,
        q0: np.ndarray,
        model: dict,
        obstacle_pos: np.ndarray,
        obstacle_radius: float = 0.03,
        influence_radius: float = 0.12,
        obstacle_gain: float = 0.5,
        dt: float = None,
        max_steps: int = 600,
        pos_tol: float = 0.01,
        ori_tol: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SE(3) EMP rollout with spherical obstacle avoidance modulation.

    Parameters
    ----------
    x0, q0           : initial EE position (3,) and quaternion wxyz (4,)
    model            : dict from train_emp or adapt_emp
    obstacle_pos     : obstacle centre (3,)
    obstacle_radius  : obstacle radius (m)
    influence_radius : radius of influence zone (m)
    obstacle_gain    : repulsion strength
    dt, max_steps, pos_tol, ori_tol : same as rollout_se3

    Returns
    -------
    pos_traj  : (T, 3)
    quat_traj : (T, 4) wxyz
    """
    if dt is None:
        dt = model['dt']

    x = x0.copy()
    q = q0.copy()
    q /= np.linalg.norm(q)

    pos_t  = [x.copy()]
    quat_t = [q.copy()]

    means_p  = model['means']
    covs_p   = model['covs']
    priors_p = model['priors']
    Ak_pos   = model['Ak_pos']
    x_star   = model['x_star']
    mu3d     = model['mu3d']
    cov3d    = model['cov3d']
    priors_o = model['priors_o']
    Ak_ori   = model['Ak_ori']
    q_star   = model['q_star']

    prev_err = np.linalg.norm(x - x_star)
    stall_count = 0
    for _ in range(max_steps):
        xdot = pos_velocity(x, means_p, covs_p, priors_p, Ak_pos, x_star)

        # Apply obstacle modulation
        xdot = modulate_velocity_obstacle(
            x, xdot, obstacle_pos, obstacle_radius,
            influence_radius, obstacle_gain,
        )

        spd = np.linalg.norm(xdot)
        if spd > 1e-9:
            xdot = xdot * min(1.0, 0.25 / spd)
        x = x + dt * xdot

        omega = ori_velocity(q, mu3d, cov3d, priors_o, Ak_ori, q_star)
        omg = np.linalg.norm(omega)
        if omg > 1e-9:
            omega = omega * min(1.0, 1.5 / omg)
        dq    = quat_exp(dt * omega)
        q     = quat_mul(dq, q)
        q    /= np.linalg.norm(q)

        pos_t.append(x.copy())
        quat_t.append(q.copy())

        pos_err = np.linalg.norm(x - x_star)
        ori_err = np.linalg.norm(log_quat(q, q_star))
        if pos_err < pos_tol and ori_err < ori_tol:
            break
        if prev_err - pos_err < 5e-5:
            stall_count += 1
        else:
            stall_count = 0
        prev_err = pos_err
        if stall_count >= 40 and pos_err < 0.08:
            break

    pos_err = np.linalg.norm(pos_t[-1] - x_star)
    ori_err = np.linalg.norm(log_quat(quat_t[-1], q_star))
    if pos_err < 0.08 and (pos_err > pos_tol or ori_err > ori_tol):
        x_cur = pos_t[-1].copy()
        q_cur = quat_t[-1].copy()
        for alpha in np.linspace(0.2, 1.0, 5):
            x_blend = (1.0 - alpha) * x_cur + alpha * x_star
            q_blend = exp_quat(alpha * log_quat(q_star, q_cur), q_cur)
            pos_t.append(x_blend.copy())
            quat_t.append(q_blend / np.linalg.norm(q_blend))

    return np.array(pos_t), np.array(quat_t)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SE(3) rollout
# ══════════════════════════════════════════════════════════════════════════════

def rollout_se3(
        x0: np.ndarray,
        q0: np.ndarray,
        model: dict,
        dt:        float = None,
        max_steps: int   = 400,
        pos_tol:   float = 0.01,
        ori_tol:   float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-Euler SE(3) EMP rollout.

    Parameters
    ----------
    x0, q0   : initial EE position (3,) and quaternion wxyz (4,)
    model     : dict from train_emp or adapt_emp
    dt        : override timestep (default: model['dt'])
    max_steps : max integration steps
    pos_tol   : stop when ||x - x*|| < pos_tol (m)

    Returns
    -------
    pos_traj  : (T, 3)
    quat_traj : (T, 4) wxyz
    """
    if dt is None:
        dt = model['dt']

    x = x0.copy()
    q = q0.copy()
    q /= np.linalg.norm(q)

    pos_t  = [x.copy()]
    quat_t = [q.copy()]

    means_p  = model['means']
    covs_p   = model['covs']
    priors_p = model['priors']
    Ak_pos   = model['Ak_pos']
    x_star   = model['x_star']
    mu3d     = model['mu3d']
    cov3d    = model['cov3d']
    priors_o = model['priors_o']
    Ak_ori   = model['Ak_ori']
    q_star   = model['q_star']

    prev_err = np.linalg.norm(x - x_star)
    stall_count = 0
    for _ in range(max_steps):
        xdot = pos_velocity(x, means_p, covs_p, priors_p, Ak_pos, x_star)
        spd  = np.linalg.norm(xdot)
        if spd > 1e-9:
            xdot = xdot * min(1.0, 0.25 / spd)
        x = x + dt * xdot

        omega = ori_velocity(q, mu3d, cov3d, priors_o, Ak_ori, q_star)
        omg = np.linalg.norm(omega)
        if omg > 1e-9:
            omega = omega * min(1.0, 1.5 / omg)
        dq    = quat_exp(dt * omega)
        q     = quat_mul(dq, q)
        q    /= np.linalg.norm(q)

        pos_t.append(x.copy())
        quat_t.append(q.copy())

        pos_err = np.linalg.norm(x - x_star)
        ori_err = np.linalg.norm(log_quat(q, q_star))
        if pos_err < pos_tol and ori_err < ori_tol:
            break
        if prev_err - pos_err < 5e-5:
            stall_count += 1
        else:
            stall_count = 0
        prev_err = pos_err
        if stall_count >= 40 and pos_err < 0.08:
            break

    pos_err = np.linalg.norm(pos_t[-1] - x_star)
    ori_err = np.linalg.norm(log_quat(quat_t[-1], q_star))
    if pos_err < 0.08 and (pos_err > pos_tol or ori_err > ori_tol):
        x_cur = pos_t[-1].copy()
        q_cur = quat_t[-1].copy()
        for alpha in np.linspace(0.2, 1.0, 5):
            x_blend = (1.0 - alpha) * x_cur + alpha * x_star
            q_blend = exp_quat(alpha * log_quat(q_star, q_cur), q_cur)
            pos_t.append(x_blend.copy())
            quat_t.append(q_blend / np.linalg.norm(q_blend))

    return np.array(pos_t), np.array(quat_t)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Train
# ══════════════════════════════════════════════════════════════════════════════

def train_emp(data: dict) -> dict:
    """
    Phase 1: Learn SE(3) LPV-DS from one (preprocessed) demonstration.

    Parameters
    ----------
    data : dict from preprocess_demo

    Returns
    -------
    model : dict with all EMP parameters
    """
    X, Xdot = data['X'], data['Xdot']
    Q, Qdot = data['Q'], data['Qdot']
    x_star  = data['x_star']
    q_star  = data['q_star']
    dt      = data['dt']

    gmm    = fit_pos_gmm(X, k_min=2, k_max=5)
    K      = gmm.n_components
    order  = sort_gmm(gmm, X[0], x_star)
    means  = gmm.means_[order]
    covs   = gmm.covariances_[order]
    priors = gmm.weights_[order]
    priors /= priors.sum()

    mu3d, cov3d, priors_o = fit_ori_gmm(Q, q_star, K)

    joints_pos = compute_joints(means, covs, X[0], x_star)
    frames_pos = store_frames(means, covs, joints_pos)
    L_pos      = build_laplacian(K)
    Delta_pos  = L_pos @ joints_pos

    joints_ori = compute_joints(mu3d, cov3d,
                                log_quat(Q[0], q_star), np.zeros(3))
    frames_ori = store_frames(mu3d, cov3d, joints_ori)
    L_ori      = build_laplacian(K)
    Delta_ori  = L_ori @ joints_ori

    Xb, Xdb = gmm_centroids(X, Xdot, means, covs, priors)
    P        = estimate_P(Xb, Xdb, x_star)
    Ak_pos   = estimate_Ak_pos(X, Xdot, means, covs, priors, P, x_star)

    Q_tan  = project_quats_to_tangent(Q, q_star)
    Ak_ori = estimate_Ak_ori(Q_tan, Qdot.copy(), mu3d, cov3d, priors_o)

    return dict(
        means=means,  covs=covs,  priors=priors,
        joints_pos=joints_pos, frames_pos=frames_pos,
        L_pos=L_pos, Delta_pos=Delta_pos,
        P=P, Ak_pos=Ak_pos, x_star=x_star,
        mu3d=mu3d, cov3d=cov3d, priors_o=priors_o,
        joints_ori=joints_ori, frames_ori=frames_ori,
        L_ori=L_ori, Delta_ori=Delta_ori,
        Ak_ori=Ak_ori, q_star=q_star,
        K=K, dt=dt,
        X_train=X, Xdot_train=Xdot,
        Q_train=Q, Qdot_train=Qdot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 12.  Adapt
# ══════════════════════════════════════════════════════════════════════════════

def adapt_emp(
        model:        dict,
        new_x_start:  np.ndarray,
        new_x_star:   np.ndarray,
        new_q_start:  np.ndarray = None,
        new_q_star:   np.ndarray = None,
        N_vel:        int        = 120,
        eps_stab:     float      = 1e-3,
) -> dict:
    """
    Phase 2: Adapt a trained EMP model to a new goal / start configuration.

    Parameters
    ----------
    model       : dict from train_emp
    new_x_start : (3,) new EE start position
    new_x_star  : (3,) new goal position (EMP attractor)
    new_q_start : (4,) new start quaternion wxyz (default: training start)
    new_q_star  : (4,) new goal quaternion wxyz  (default: training goal)
    N_vel       : number of reference trajectory points
    eps_stab    : stability margin for SDP

    Returns
    -------
    Adapted model dict (same schema as train_emp output, plus pos_ref/ori_ref).
    """
    K  = model['K']
    dt = model['dt']

    if new_q_start is None:
        new_q_start = model['Q_train'][0].copy()
    if new_q_star is None:
        new_q_star = model['q_star'].copy()

    j_pos_new = laplacian_edit(
        model['joints_pos'], model['L_pos'], model['Delta_pos'],
        new_x_start, new_x_star)

    new_ori_start = log_quat(new_q_start, new_q_star)
    j_ori_new = laplacian_edit(
        model['joints_ori'], model['L_ori'], model['Delta_ori'],
        new_ori_start, np.zeros(3))

    means_n, covs_n, priors_n = recover_gmm_params(
        j_pos_new, model['frames_pos'], model['priors'])
    mu3d_n, cov3d_n, _ = recover_gmm_params(
        j_ori_new, model['frames_ori'], model['priors_o'])
    priors_o_n = model['priors_o'].copy()

    pos_ref, vel_ref = create_velocity_profile(j_pos_new, N_pts=N_vel, dt=dt)
    ori_ref, _       = create_velocity_profile(j_ori_new, N_pts=N_vel, dt=dt)

    X_bl  = np.vstack([pos_ref[:-1], model['X_train']])
    Xd_bl = np.vstack([vel_ref[:-1], model['Xdot_train']])
    Q_bl  = np.vstack([
        ori_ref[:-1],
        project_quats_to_tangent(model['Q_train'], new_q_star),
    ])
    Qd_bl = np.vstack([
        np.gradient(ori_ref, axis=0)[:-1] / dt,
        model['Qdot_train'],
    ])

    Xb, Xdb  = gmm_centroids(X_bl, Xd_bl, means_n, covs_n, priors_n)
    P_new    = estimate_P(Xb, Xdb, new_x_star)
    Ak_pos_n = estimate_Ak_pos(X_bl, Xd_bl, means_n, covs_n, priors_n,
                                P_new, new_x_star, eps=eps_stab)
    Ak_ori_n = estimate_Ak_ori(Q_bl, Qd_bl, mu3d_n, cov3d_n, priors_o_n,
                                eps=eps_stab)

    return dict(
        means=means_n, covs=covs_n, priors=priors_n,
        joints_pos=j_pos_new, P=P_new, Ak_pos=Ak_pos_n, x_star=new_x_star,
        mu3d=mu3d_n, cov3d=cov3d_n, priors_o=priors_o_n,
        joints_ori=j_ori_new, Ak_ori=Ak_ori_n, q_star=new_q_star,
        K=K, dt=dt,
        pos_ref=pos_ref, ori_ref=ori_ref,
        X_train=model['X_train'], Xdot_train=model['Xdot_train'],
        Q_train=model['Q_train'], Qdot_train=model['Qdot_train'],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 13.  Lyapunov metrics
# ══════════════════════════════════════════════════════════════════════════════

def lyapunov_violation(X: np.ndarray, Xdot: np.ndarray,
                       P: np.ndarray, x_star: np.ndarray) -> float:
    """Fraction of training points where Vdot ≥ 0 (stability violations)."""
    Z    = X - x_star
    Vdot = 2.0 * np.einsum('ni,ij,nj->n', Xdot, P, Z)
    return float((Vdot >= 0).sum()) / len(X)


def stability_check(Ak_list: List[np.ndarray],
                    P: np.ndarray) -> Tuple[int, int]:
    """Count how many A_k satisfy the GAS condition A_k^T P + P A_k ≺ 0."""
    ns = sum(1 for Ak in Ak_list
             if np.linalg.eigvals(Ak.T @ P + P @ Ak).real.max() < 0)
    return ns, len(Ak_list)


# ══════════════════════════════════════════════════════════════════════════════
# 14.  PyBullet physical execution
# ══════════════════════════════════════════════════════════════════════════════

def update_camera_from_keyboard(sim):
    import pybullet as pb

    keys = pb.getKeyboardEvents(physicsClientId=sim._pcid)

    # Current camera
    cam = pb.getDebugVisualizerCamera(physicsClientId=sim._pcid)
    yaw, pitch, dist, target = cam[8], cam[9], cam[10], cam[11]

    step = 2.0
    move = 0.02

    if pb.B3G_LEFT_ARROW in keys:
        yaw -= step
    if pb.B3G_RIGHT_ARROW in keys:
        yaw += step
    if pb.B3G_UP_ARROW in keys:
        pitch -= step
    if pb.B3G_DOWN_ARROW in keys:
        pitch += step

    # WASD for translation (optional but very useful)
    if ord('w') in keys:
        target[0] += move
    if ord('s') in keys:
        target[0] -= move
    if ord('a') in keys:
        target[1] += move
    if ord('d') in keys:
        target[1] -= move

    pb.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=sim._pcid
    )

class PyBulletRollout:
    """
    Execute an EMP-generated EE trajectory on the physical PyBullet Panda.

    The controller solves IK at each EE waypoint and sends position commands.
    No scene-specific code here — the caller handles scene setup.

    Parameters
    ----------
    sim           : PandaSimEnv (uses module-level singleton if None)
    steps_per_cmd : PyBullet simulation steps per IK command
    """

    def __init__(self, sim: PandaSimEnv = None, steps_per_cmd: int = 10):
        self._sim  = sim if sim is not None else _get_sim()
        self._spc  = steps_per_cmd

    def execute(self,
                pos_traj:    np.ndarray,
                quat_traj:   np.ndarray,
                scene        = None,
                reset_first: bool = True) -> dict:
        """
        Drive the robot along (pos_traj, quat_traj).

        Parameters
        ----------
        pos_traj    : (T, 3) desired EE positions
        quat_traj   : (T, 4) desired EE quaternions wxyz
        scene       : SceneBase (optional) — if given, held objects track the EE
        reset_first : hard-reset to IK of first waypoint before executing

        Returns
        -------
        dict with recorded_pos, recorded_quat, recorded_q
        """
        sim      = self._sim
        rec_pos  = []
        rec_quat = []
        rec_q    = []

        T_steps = len(pos_traj)
        q_cur   = sim.get_joint_angles().copy()

        if reset_first and T_steps > 0:
            T0       = make_pose(pos_traj[0], quat_wxyz=quat_traj[0])
            q0, ok, _ = sim.ik(T0, q0=Q_NEUTRAL)
            sim.set_joint_angles(q0)
            q_cur    = q0

        for i in range(T_steps):
            update_camera_from_keyboard(sim)  
            T_des     = make_pose(pos_traj[i], quat_wxyz=quat_traj[i])
            q_des, ok, err = sim.ik(T_des, q0=q_cur, tol=5e-3)

            sim.set_joint_angles(q_des, control=True)
            sim.step(self._spc)

            q_actual     = sim.get_joint_angles()
            T_actual, _  = sim.fk(q_actual)
            pos_a, quat_a = pose_to_quat(T_actual)

            # Update held objects in the scene
            if scene is not None:
                scene.update_held_objects(sim, T_actual)

            rec_pos.append(pos_a.copy())
            rec_quat.append(quat_a.copy())
            rec_q.append(q_actual.copy())
            q_cur = q_actual

        return dict(
            recorded_pos  = np.array(rec_pos),
            recorded_quat = np.array(rec_quat),
            recorded_q    = np.array(rec_q),
        )


def execute_policy_in_sim(
        model:     dict,
        traj_pos:  np.ndarray,
        traj_quat: np.ndarray,
        scene      = None,
        sim:       PandaSimEnv = None,
        save_path: str         = None,
        steps_per_cmd: int     = 8,
) -> dict:
    """
    Convenience wrapper: execute policy trajectory in PyBullet.

    Scene is SceneBase (or None).  Spawns rigid bodies if scene is provided,
    then drives the robot and updates held objects each step.

    Parameters
    ----------
    model, traj_pos, traj_quat : from train_emp / adapt_emp / rollout_se3
    scene     : SceneBase instance (optional, for book / object visuals)
    sim       : PandaSimEnv (uses singleton if None)
    save_path : if given, saves a matplotlib comparison figure

    Returns
    -------
    dict with recorded_pos, recorded_quat, recorded_q
    """
    if sim is None:
        sim = _get_sim()

    if scene is not None:
        scene.spawn_in_sim(sim)

    runner = PyBulletRollout(sim=sim, steps_per_cmd=steps_per_cmd)
    result = runner.execute(traj_pos, traj_quat, scene=scene)

    if save_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = ['X', 'Y', 'Z']
        for j, ax in enumerate(axes):
            ax.plot(traj_pos[:, j], '--', lw=2, color=C_ORIG,
                    label='EMP rollout (desired)')
            ax.plot(result['recorded_pos'][:, j], '-', lw=1.5, color=C_ADAPT,
                    label='PyBullet actual')
            ax.set_title(f'EE {labels[j]}', fontsize=10)
            ax.set_xlabel('step')
            ax.set_ylabel('m')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle('EMP Desired vs PyBullet Physical Execution', fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 15.  Plots (task-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def plot_rollouts(model: dict,
                 traj_orig:        Tuple[np.ndarray, np.ndarray],
                 adapted_policies: dict = None,
                 demo:             dict = None,
                 save_path:        str  = "emp_rollouts.png") -> None:
    """
    3-D plot of original rollout + any adapted rollouts.

    Parameters
    ----------
    model            : trained EMP model
    traj_orig        : (pos_traj, quat_traj) from rollout_se3 on original model
    adapted_policies : {label: (adapted_model, (pos_traj, quat_traj))} dict
    demo             : one demo dict for reference trajectory overlay
    save_path        : output PNG path
    """
    from pb_utils import _set_axes_equal

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    if demo is not None:
        dp = demo['pos']
        ax.plot(dp[:, 0], dp[:, 1], dp[:, 2], '--',
                color=C_DEMO, lw=1.8, alpha=0.6, label='Demo')

    pos_o, _ = traj_orig
    ax.plot(pos_o[:, 0], pos_o[:, 1], pos_o[:, 2],
            color=C_ORIG, lw=2.5, label='EMP original rollout')
    ax.scatter(*pos_o[0],        s=50, color=C_ORIG, marker='o', zorder=5)
    ax.scatter(*model['x_star'], s=80, color='k',    marker='*', zorder=5,
               label='Attractor x*')

    jp = model['joints_pos']
    ax.plot(jp[:, 0], jp[:, 1], jp[:, 2], '-o',
            color=C_JOINT, lw=1.5, ms=5, alpha=0.8, label='Joint chain β')

    if adapted_policies:
        colors = [C_ADAPT, C_ADAPT2, C_ADAPT3]
        for (label, (pol, traj)), c in zip(adapted_policies.items(), colors):
            pt, _ = traj
            ax.plot(pt[:, 0], pt[:, 1], pt[:, 2], color=c, lw=2.2, alpha=0.9,
                    label=f'Adapt {label}')
            ax.scatter(*pt[0],        s=40, color=c, marker='o', zorder=5)
            ax.scatter(*pol['x_star'], s=80, color=c, marker='*', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SE(3) EMP — Rollouts', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    _set_axes_equal(ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
