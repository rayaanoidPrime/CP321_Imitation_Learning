"""
pb_stage3_emp.py — SE(3) EMP: Training, Adaptation, Validation (PyBullet)
==========================================================================
Drop-in replacement for stage3_emp.py.

Changes from original:
  • Imports from pb_stage1_env / pb_stage2_demos (PyBullet FK/IK)
  • Adds PyBulletRollout class — physically executes EMP trajectories
  • Adds execute_policy_in_sim() — drives the Panda arm via joint commands
  • All original EMP math is preserved verbatim
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as Rot
from sklearn.mixture import GaussianMixture
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
warnings.filterwarnings('ignore')

# ── Drop-in replace ──────────────────────────────────────────────────────────
from pb_stage1_env import (
    fk, ik, Q_NEUTRAL, JOINT_LIMITS,
    BookRackScene, make_pose, pose_to_quat,
    se3_interpolate, quat_log, quat_exp, quat_conjugate, quat_mul,
    draw_arm, draw_frame, draw_box, _set_axes_equal,
    _get_sim, PandaSimEnv,
)
from pb_stage2_demos import load_demos, generate_demonstrations, save_demos

np.random.seed(0)

# Colour palette
C_DEMO   = "#3B8BD4"
C_ORIG   = "#1D9E75"
C_ADAPT  = "#D85A30"
C_ADAPT2 = "#9933CC"
C_ADAPT3 = "#CC3366"
C_JOINT  = "#BA7517"


# ==============================================================================
# 3.1  Data preprocessing
# ==============================================================================

def preprocess_demo(demo: dict, demo_idx: int = 0, N_max: int = 150) -> dict:
    """Extract position + quaternion training arrays from one demonstration."""
    pos  = demo['pos']
    quat = demo['quat']
    t    = demo['times']
    dt   = float(np.diff(t).mean())

    idx  = np.round(np.linspace(0, len(pos)-1,
                                min(N_max, len(pos)))).astype(int)
    pos  = pos[idx]
    quat = quat[idx]
    t    = t[idx]

    Xdot = np.gradient(pos,  axis=0) / dt
    dqdt = np.gradient(quat, axis=0) / dt
    Qdot = np.zeros((len(quat), 3))
    for i in range(len(quat)):
        qc       = quat_conjugate(quat[i])
        prod     = quat_mul(2*dqdt[i], qc)
        Qdot[i]  = prod[1:]

    x_star = pos[-1].copy()
    q_star = quat[-1].copy()

    return dict(X=pos[:-1], Xdot=Xdot[:-1],
                Q=quat[:-1], Qdot=Qdot[:-1],
                x_star=x_star, q_star=q_star, dt=dt)


# ==============================================================================
# 3.2  Position GMM
# ==============================================================================

def fit_pos_gmm(X, k_min=2, k_max=6, n_init=5):
    best, bic = None, np.inf
    for k in range(k_min, k_max+1):
        g = GaussianMixture(k, covariance_type='full',
                            n_init=n_init, random_state=0)
        g.fit(X)
        b = g.bic(X)
        if b < bic:
            bic, best = b, g
    return best


def sort_gmm(gmm, x_start, x_end):
    d = x_end - x_start
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.arange(gmm.n_components)
    return np.argsort(gmm.means_ @ (d/n))


def gamma_pos_batch(X, means, covs, priors):
    N, K = len(X), len(means)
    lp   = np.zeros((N, K))
    for k in range(K):
        diff = X - means[k]
        ci   = np.linalg.inv(covs[k])
        lds  = np.linalg.slogdet(covs[k])[1]
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:,k] = np.log(priors[k]+1e-300) - 0.5*(3*np.log(2*np.pi)+lds+maha)
    lp -= lp.max(1, keepdims=True)
    p   = np.exp(lp)
    return p / (p.sum(1, keepdims=True) + 1e-300)


def gamma_pos_single(x, means, covs, priors):
    return gamma_pos_batch(x[None], means, covs, priors)[0]


# ==============================================================================
# 3.3  Orientation (Quaternion) GMM on tangent plane
# ==============================================================================

def quat_null_basis(q_wxyz):
    q = q_wxyz / np.linalg.norm(q_wxyz)
    A = np.eye(4) - np.outer(q, q)
    U, _, _ = np.linalg.svd(A)
    return U[:, :3]


def log_quat(q_wxyz, q_att_wxyz):
    qa  = q_att_wxyz / np.linalg.norm(q_att_wxyz)
    qc  = quat_conjugate(qa)
    q_rel = quat_mul(qc, q_wxyz / np.linalg.norm(q_wxyz))
    w = np.clip(q_rel[0], -1, 1)
    v = q_rel[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-10:
        return np.zeros(3)
    return (np.arccos(w) / nv) * v


def exp_quat(omega, q_att_wxyz):
    qa  = q_att_wxyz / np.linalg.norm(q_att_wxyz)
    q_d = quat_exp(omega)
    return quat_mul(qa, q_d)


def project_quats_to_tangent(Q, q_att):
    return np.array([log_quat(Q[i], q_att) for i in range(len(Q))])


def fit_ori_gmm(Q, q_star, K):
    Q_tan = project_quats_to_tangent(Q, q_star)
    g = GaussianMixture(K, covariance_type='full', n_init=5, random_state=0)
    g.fit(Q_tan)
    return g.means_, g.covariances_, g.weights_


def gamma_ori_batch(Q, mu3d, cov3d, priors, q_star):
    Q_tan = project_quats_to_tangent(Q, q_star)
    N, K  = len(Q_tan), len(mu3d)
    lp    = np.zeros((N, K))
    for k in range(K):
        diff = Q_tan - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6*np.eye(3))
        lds  = np.linalg.slogdet(cov3d[k] + 1e-6*np.eye(3))[1]
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:,k] = np.log(priors[k]+1e-300) - 0.5*(3*np.log(2*np.pi)+lds+maha)
    lp -= lp.max(1, keepdims=True)
    p   = np.exp(lp)
    return p / (p.sum(1, keepdims=True) + 1e-300)


# ==============================================================================
# 3.4  Joint chain & local-frame storage
# ==============================================================================

def compute_joints(means, covs, x_start, x_end):
    K = len(means)
    interior = []
    for k in range(K-1):
        Si  = np.linalg.inv(covs[k]   + 1e-8*np.eye(3))
        Si1 = np.linalg.inv(covs[k+1] + 1e-8*np.eye(3))
        St  = np.linalg.inv(Si + Si1)
        interior.append(St @ (Si @ means[k] + Si1 @ means[k+1]))
    rows = [x_start[None]] + ([np.array(interior)] if interior else []) + [x_end[None]]
    return np.vstack(rows)


def local_frame(a, b):
    e = b - a; d = np.linalg.norm(e)
    if d < 1e-12:
        return np.eye(3), 0.
    e /= d
    t = np.array([1.,0.,0.]) if abs(e[0]) < 0.9 else np.array([0.,1.,0.])
    t -= e * (e @ t); t /= np.linalg.norm(t)
    return np.column_stack([e, t, np.cross(e, t)]), d


def store_frames(means, covs, joints):
    frames = []
    for k in range(len(means)):
        R, d    = local_frame(joints[k], joints[k+1])
        mu_loc  = R.T @ (means[k] - joints[k])
        ev, E   = np.linalg.eigh(covs[k])
        E_loc   = R.T @ E
        frames.append(dict(mu_loc=mu_loc, E_loc=E_loc, eigvals=ev, dist=d))
    return frames


def recover_gmm_params(joints_new, frames, priors):
    K = len(frames); means_n, covs_n = [], []
    for k in range(K):
        R_new, d_new = local_frame(joints_new[k], joints_new[k+1])
        s   = d_new / frames[k]['dist'] if frames[k]['dist'] > 1e-12 else 1.0
        mu  = joints_new[k] + R_new @ frames[k]['mu_loc']
        E   = R_new @ frames[k]['E_loc']
        ev  = np.maximum(frames[k]['eigvals'] * s, 1e-8)
        cov = E @ np.diag(ev) @ E.T
        means_n.append(mu)
        covs_n.append(0.5*(cov+cov.T))
    return np.array(means_n), np.array(covs_n), priors.copy()


# ==============================================================================
# 3.5  Laplacian editing
# ==============================================================================

def build_laplacian(n):
    L = np.zeros((n+1, n+1))
    for i in range(n+1):
        if i == 0:   L[0,0]=1.;   L[0,1]=-1.
        elif i == n: L[n,n]=1.;   L[n,n-1]=-1.
        else:        L[i,i]=1.;   L[i,i-1]=L[i,i+1]=-0.5
    return L


def laplacian_edit(joints, L, Delta, new_start, new_end):
    n = len(joints)-1
    if n == 1:
        return np.vstack([new_start, new_end])
    L_int = L[1:n, 1:n]
    rhs   = Delta[1:n] - np.outer(L[1:n,0], new_start) - np.outer(L[1:n,n], new_end)
    beta  = np.linalg.solve(L_int, rhs)
    return np.vstack([new_start, beta, new_end])


# ==============================================================================
# 3.6  Velocity profile regeneration
# ==============================================================================

def create_velocity_profile(joints, N_pts=150, dt=0.01):
    segs  = np.diff(joints, axis=0)
    dists = np.linalg.norm(segs, axis=1)
    total = dists.sum()
    if total < 1e-12:
        return np.tile(joints[0], (N_pts, 1)), np.zeros((N_pts, joints.shape[1]))
    cum  = np.concatenate([[0.], np.cumsum(dists)])
    lam  = cum / total
    idx_j = np.clip(np.round(lam*(N_pts-1)).astype(int), 0, N_pts-1)
    t_u   = np.linspace(0, 1, N_pts)
    pos   = np.column_stack([np.interp(t_u, lam, joints[:,d])
                              for d in range(joints.shape[1])])
    L   = build_laplacian(N_pts-1)
    Delta = L @ pos
    A_a  = np.vstack([L, np.eye(N_pts)[idx_j]])
    b_a  = np.vstack([Delta, joints])
    pos  = np.linalg.lstsq(A_a.T@A_a, A_a.T@b_a, rcond=None)[0]
    vel  = np.gradient(pos, axis=0) / dt
    return pos, vel


# ==============================================================================
# 3.7  Lyapunov P estimation
# ==============================================================================

def gmm_centroids(X, Xdot, means, covs, priors):
    g      = gamma_pos_batch(X, means, covs, priors)
    assign = np.argmax(g, axis=1)
    K = len(means)
    Xb, Xdb = [], []
    for k in range(K):
        m = assign == k
        Xb.append(X[m].mean(0)    if m.sum() > 0 else means[k])
        Xdb.append(Xdot[m].mean(0) if m.sum() > 0 else np.zeros(3))
    return np.array(Xb), np.array(Xdb)


def estimate_P(X_bar, Xdot_bar, x_star, eps=1e-5):
    d, K = 3, len(X_bar)
    Z    = X_bar - x_star
    P_v  = cp.Variable((d, d), symmetric=True)
    s    = cp.Variable(K, nonneg=True)
    cons = [P_v >> float(eps)*np.eye(d)]
    for k in range(K):
        cons.append(s[k] >= Xdot_bar[k] @ P_v @ Z[k])
    cp.Problem(cp.Minimize(cp.sum(s)), cons).solve(
        solver=cp.SCS, verbose=False, eps=1e-5)
    Pv = P_v.value
    if Pv is None or not np.all(np.isfinite(Pv)):
        Pv = np.eye(d)
    Pv = 0.5*(Pv + Pv.T)
    ev = float(np.linalg.eigvalsh(Pv).min())
    ep = float(np.asarray(eps).flat[0])
    if ev < ep:
        Pv += (ep - ev + 1e-8) * np.eye(d)
    return Pv


# ==============================================================================
# 3.8  Stability-constrained SDP for {A_k}
# ==============================================================================

def project_stable_pos(Ak, P, eps=1e-3):
    M   = Ak.T @ P + P @ Ak
    lam = float(np.linalg.eigvalsh(M).max())
    if lam >= -eps:
        alpha = (lam + 2*eps) / (2 * float(np.linalg.eigvalsh(P).min()))
        Ak    = Ak - alpha * np.eye(Ak.shape[0])
    return Ak


def project_stable_ori(Ak, eps=1e-3):
    lam = float(np.linalg.eigvalsh(Ak + Ak.T).max())
    if lam >= -eps:
        Ak = Ak - ((lam + 2*eps) / 2) * np.eye(Ak.shape[0])
    return Ak


def estimate_Ak_pos(X, Xdot, means, covs, priors, P, x_star, eps=1e-3):
    K_l  = len(means); d = 3
    Z    = X - x_star
    g    = gamma_pos_batch(X, means, covs, priors)
    Av   = [cp.Variable((d, d)) for _ in range(K_l)]
    pred = sum(cp.multiply(g[:,k:k+1], Z @ Av[k].T) for k in range(K_l))
    obj  = cp.Minimize(cp.sum_squares(Xdot - pred) / len(X))
    cons = [Av[k].T @ P + P @ Av[k] << -float(eps)*np.eye(d)
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


def estimate_Ak_ori(Q_tan, Qdot_tan, mu3d, cov3d, priors, eps=1e-3):
    K_l = len(mu3d); d = 3; N = len(Q_tan)
    g   = np.zeros((N, K_l))
    lp = np.zeros((N, K_l))
    for k in range(K_l):
        diff = Q_tan - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6*np.eye(3))
        maha = np.einsum('ni,ij,nj->n', diff, ci, diff)
        lp[:,k] = np.log(priors[k]+1e-300) - 0.5*maha   # log-prob, NOT exp'd
    lp -= lp.max(1, keepdims=True)
    g = np.exp(lp)                                        # exp exactly once
    g /= g.sum(1, keepdims=True) + 1e-300


    Av   = [cp.Variable((d, d)) for _ in range(K_l)]
    pred = sum(cp.multiply(g[:,k:k+1], Q_tan @ Av[k].T) for k in range(K_l))
    obj  = cp.Minimize(cp.sum_squares(Qdot_tan - pred) / N)
    cons = [Av[k] + Av[k].T << -float(eps)*np.eye(d) for k in range(K_l)]
    cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False,
                                eps=1e-4, max_iters=10000)
    result = []
    for k in range(K_l):
        Ak = Av[k].value
        if Ak is None or not np.all(np.isfinite(Ak)):
            Ak = -np.eye(d)
        result.append(project_stable_ori(Ak, eps=float(eps)))
    return result


# ==============================================================================
# 3.9  Runtime velocity computation
# ==============================================================================

def pos_velocity(x, means, covs, priors, Ak_pos, x_star, g_ood=1.5):
    gamma = gamma_pos_single(x, means, covs, priors)
    z     = x - x_star
    v     = sum(gamma[k] * (Ak_pos[k] @ z) for k in range(len(Ak_pos)))
    g     = g_ood * (1.0 - float(gamma.max()))
    return v - g*z


def ori_velocity(q, mu3d, cov3d, priors, Ak_ori, q_star):
    K   = len(mu3d)
    log_q = log_quat(q, q_star)
    g     = np.zeros(K)
    for k in range(K):
        diff = log_q - mu3d[k]
        ci   = np.linalg.inv(cov3d[k] + 1e-6*np.eye(3))
        g[k] = np.exp(-0.5*(diff@ci@diff) + np.log(priors[k]+1e-300))
    g /= g.sum() + 1e-300
    return sum(g[k] * (Ak_ori[k] @ log_q) for k in range(K))


# ==============================================================================
# 3.10  Full SE(3) rollout (EMP)
# ==============================================================================

def rollout_se3(x0, q0,
                means_p, covs_p, priors_p, Ak_pos, x_star,
                mu3d,    cov3d,  priors_o, Ak_ori, q_star,
                dt=0.02, max_steps=800, pos_tol=0.04):
    """
    Forward-Euler SE(3) rollout.
    Returns (pos_traj (T,3), quat_traj (T,4) wxyz).
    """
    x  = x0.copy(); q = q0.copy()
    q /= np.linalg.norm(q)
    pos_t = [x.copy()]; quat_t = [q.copy()]

    for _ in range(max_steps):
        xdot = pos_velocity(x, means_p, covs_p, priors_p, Ak_pos, x_star)
        spd  = np.linalg.norm(xdot)
        if spd > 1e-9:
            xdot = xdot * min(1.0, 0.3/spd)
        x = x + dt * xdot

        omega = ori_velocity(q, mu3d, cov3d, priors_o, Ak_ori, q_star)
        # Parallel transport omega from tangent space at q_star to tangent space at q
        # Approximate: for small dt, use body-frame integration via left multiplication
        dq = quat_exp(dt * omega)
        q  = quat_mul(dq, q)    # left-multiply = apply in world frame (closer to paper Eq. 20)
        q /= np.linalg.norm(q)

        pos_t.append(x.copy()); quat_t.append(q.copy())
        if np.linalg.norm(x - x_star) < pos_tol:
            break

    return np.array(pos_t), np.array(quat_t)


# ==============================================================================
# 3.11  Train EMP
# ==============================================================================

def train_emp(data: dict) -> dict:
    """Phase 1: Learn SE(3) LPV-DS from one demonstration."""
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
    priors = gmm.weights_[order]; priors /= priors.sum()

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

    Q_tan    = project_quats_to_tangent(Q, q_star)
    Ak_ori   = estimate_Ak_ori(Q_tan, Qdot.copy(), mu3d, cov3d, priors_o)

    return dict(
        means=means, covs=covs, priors=priors,
        joints_pos=joints_pos, frames_pos=frames_pos,
        L_pos=L_pos, Delta_pos=Delta_pos,
        P=P, Ak_pos=Ak_pos, x_star=x_star,
        mu3d=mu3d, cov3d=cov3d, priors_o=priors_o,
        joints_ori=joints_ori, frames_ori=frames_ori,
        L_ori=L_ori, Delta_ori=Delta_ori,
        Ak_ori=Ak_ori, q_star=q_star,
        K=K, dt=dt, X_train=X, Xdot_train=Xdot,
        Q_train=Q, Qdot_train=Qdot,
    )


# ==============================================================================
# 3.12  Adapt EMP
# ==============================================================================

def adapt_emp(model: dict,
              new_x_start: np.ndarray,
              new_x_star:  np.ndarray,
              new_q_start: np.ndarray = None,
              new_q_star:  np.ndarray = None,
              N_vel:       int        = 120,
              eps_stab:    float      = 1e-3) -> dict:
    """Phase 2: Adapt SE(3) EMP to new scene configuration."""
    K  = model['K']
    dt = model['dt']

    if new_q_start is None: new_q_start = model['Q_train'][0].copy()
    if new_q_star  is None: new_q_star  = model['q_star'].copy()

    j_pos_new = laplacian_edit(model['joints_pos'], model['L_pos'],
                                model['Delta_pos'], new_x_start, new_x_star)

    new_ori_start = log_quat(new_q_start, new_q_star)
    j_ori_new = laplacian_edit(model['joints_ori'], model['L_ori'],
                                model['Delta_ori'],
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
    Q_bl  = np.vstack([ori_ref[:-1],
                       project_quats_to_tangent(model['Q_train'], new_q_star)])
    Qd_bl = np.vstack([np.gradient(ori_ref, axis=0)[:-1]/dt,
                        model['Qdot_train']])

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
        K=K, dt=dt, pos_ref=pos_ref, ori_ref=ori_ref,
    )


# ==============================================================================
# 3.13  Lyapunov metrics
# ==============================================================================

def lyapunov_violation(X, Xdot, P, x_star) -> float:
    Z    = X - x_star
    Vdot = 2.0 * np.einsum('ni,ij,nj->n', Xdot, P, Z)
    return float((Vdot >= 0).sum()) / len(X)


def stability_check(Ak_list, P):
    ns = sum(1 for Ak in Ak_list
             if np.linalg.eigvals(Ak.T@P + P@Ak).real.max() < 0)
    return ns, len(Ak_list)


# ==============================================================================
# 3.14  PyBullet Physical Execution  (NEW)
# ==============================================================================

class PyBulletRollout:
    """
    Execute an EMP-generated EE trajectory on the physical PyBullet Panda.

    The controller solves IK at each EE waypoint and sends joint commands
    to the simulation.  The result is a physically-simulated execution that
    respects joint limits and collisions.

    Parameters
    ----------
    sim      : PandaSimEnv (or None to use the module-level singleton)
    dt_sim   : simulation timestep per control step (number of pb steps)
    """

    def __init__(self, sim: PandaSimEnv = None, steps_per_cmd: int = 10):
        self._sim    = sim if sim is not None else _get_sim()
        self._spc    = steps_per_cmd
        self._ee_traj:    list = []   # recorded EE positions
        self._joint_traj: list = []   # recorded joint angles

    def execute(self,
                pos_traj:  np.ndarray,
                quat_traj: np.ndarray,
                reset_first: bool = True) -> dict:
        """
        Drive the robot along (pos_traj, quat_traj).

        Parameters
        ----------
        pos_traj  : (T, 3) EE positions
        quat_traj : (T, 4) EE quaternions wxyz
        reset_first : if True, hard-reset to IK of first waypoint

        Returns dict with recorded_pos, recorded_quat, recorded_q.
        """
        sim   = self._sim
        rec_pos  = []
        rec_quat = []
        rec_q    = []

        T_steps = len(pos_traj)
        q_cur   = sim.get_joint_angles().copy()

        if reset_first and T_steps > 0:
            T0 = make_pose(pos_traj[0], quat_wxyz=quat_traj[0])
            q0, ok, _ = sim.ik(T0, q0=Q_NEUTRAL)
            sim.set_joint_angles(q0)
            q_cur = q0

        for i in range(T_steps):
            T_des = make_pose(pos_traj[i], quat_wxyz=quat_traj[i])
            q_des, ok, err = sim.ik(T_des, q0=q_cur, tol=5e-3)

            # Send position command
            sim.set_joint_angles(q_des, control=True)

            # Step the physics simulation
            sim.step(self._spc)

            # Record actual state
            q_actual = sim.get_joint_angles()
            T_actual, _ = sim.fk(q_actual)
            pos_a, quat_a = pose_to_quat(T_actual)

            rec_pos.append(pos_a.copy())
            rec_quat.append(quat_a.copy())
            rec_q.append(q_actual.copy())
            q_cur = q_actual

        return dict(
            recorded_pos  = np.array(rec_pos),
            recorded_quat = np.array(rec_quat),
            recorded_q    = np.array(rec_q),
        )

    def reset(self) -> None:
        self._sim.reset()


def execute_policy_in_sim(model: dict,
                           traj_pos:  np.ndarray,
                           traj_quat: np.ndarray,
                           scene:     BookRackScene = None,
                           save_path: str            = None,
                           sim:       PandaSimEnv    = None) -> dict:
    """
    Convenience wrapper: execute policy trajectory in PyBullet and optionally
    save a matplotlib comparison figure.

    Returns recorded trajectory dict.
    """
    if sim is None:
        sim = _get_sim()

    if scene is not None:
        scene.spawn_in_sim(sim)

    runner = PyBulletRollout(sim=sim)
    result = runner.execute(traj_pos, traj_quat)

    if save_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = ['X', 'Y', 'Z']
        for j, ax in enumerate(axes):
            ax.plot(traj_pos[:,j],                  '--', lw=2,
                    color=C_ORIG,  label='EMP rollout (desired)')
            ax.plot(result['recorded_pos'][:,j],    '-',  lw=1.5,
                    color=C_ADAPT, label='PyBullet actual')
            ax.set_title(f'EE {labels[j]}', fontsize=10)
            ax.set_xlabel('step'); ax.set_ylabel('m')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle('Stage 3 — EMP Desired vs PyBullet Physical Execution',
                     fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return result


# ==============================================================================
# 3.15  Stage 3 Tests
# ==============================================================================

def run_stage3_tests(model, adapted_policies, demos, original_traj) -> tuple:
    print("=" * 65)
    print("STAGE 3 TESTS — SE(3) EMP Model (PyBullet)")
    print("=" * 65)
    passed = 0; total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total; total += 1
        sym = "✓" if cond else "✗"
        print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1

    P  = model['P']
    ev = np.linalg.eigvalsh(P)
    check("P is symmetric PD", bool(np.all(ev > 0)), f"min_eig={ev.min():.4f}")

    ns, nt = stability_check(model['Ak_pos'], model['P'])
    check("Position GAS: all A_k stable", ns == nt, f"{ns}/{nt}")

    ns_o = sum(1 for Ak in model['Ak_ori']
               if np.linalg.eigvals(Ak + Ak.T).real.max() < 0)
    nt_o = len(model['Ak_ori'])
    check("Orientation A_k^ori all ≺ 0", ns_o == nt_o, f"{ns_o}/{nt_o}")

    gmat  = gamma_pos_batch(model['X_train'], model['means'],
                             model['covs'],   model['priors'])
    Z     = model['X_train'] - model['x_star']
    Xd_pred = sum(gmat[:,k:k+1]*(Z @ model['Ak_pos'][k].T)
                  for k in range(model['K']))
    viol  = lyapunov_violation(model['X_train'], Xd_pred,
                                model['P'], model['x_star'])
    check("Lyapunov violation rate < 25%", viol < 0.25, f"{100*viol:.1f}%")

    pos_t, _ = original_traj
    fd = float(np.linalg.norm(pos_t[-1] - model['x_star']))
    check("Original rollout converges (< 10 cm)", fd < 0.10, f"dist={fd*100:.1f}cm")

    for label, (pol, traj) in adapted_policies.items():
        pos_a, _ = traj
        ns_a, nt_a = stability_check(pol['Ak_pos'], pol['P'])
        check(f"Adapt {label}: GAS ({ns_a}/{nt_a})", ns_a == nt_a)
        fd_a = float(np.linalg.norm(pos_a[-1] - pol['x_star']))
        check(f"Adapt {label}: convergence (< 10 cm)", fd_a < 0.10,
              f"dist={fd_a*100:.1f}cm")

    _, quat_t = original_traj
    ori_err = np.array([
        np.linalg.norm(log_quat(quat_t[i], model['q_star']))
        for i in range(len(quat_t))
    ])
    check("Orientation error decreasing",
          bool(ori_err[-1] < ori_err[0]),
          f"start={ori_err[0]:.3f} end={ori_err[-1]:.3f} rad")

    print(f"\n  Result: {passed}/{total} tests passed")
    return passed, total


# ==============================================================================
# 3.16  Visualisation
# ==============================================================================

def plot_se3_policy(model, traj_orig, adapted_policies, demo,
                    save_path="s3_se3_policy.png") -> None:
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    dp = demo['pos']
    ax.plot(dp[:,0], dp[:,1], dp[:,2], '--', color=C_DEMO, lw=1.8,
            alpha=0.6, label='Demo (demo 0)')

    pos_o, _ = traj_orig
    ax.plot(pos_o[:,0], pos_o[:,1], pos_o[:,2], color=C_ORIG, lw=2.5,
            label='EMP original rollout')
    ax.scatter(*pos_o[0],           s=50, color=C_ORIG,  marker='o', zorder=5)
    ax.scatter(*model['x_star'],    s=80, color='k',      marker='*', zorder=5,
               label='Attractor x*')

    jp = model['joints_pos']
    ax.plot(jp[:,0], jp[:,1], jp[:,2], '-o', color=C_JOINT,
            lw=1.5, ms=5, alpha=0.8, label='Joint chain β')

    colors = [C_ADAPT, C_ADAPT2, C_ADAPT3]
    for (label, (pol, traj)), c in zip(adapted_policies.items(), colors):
        pt, _ = traj
        ax.plot(pt[:,0], pt[:,1], pt[:,2], color=c, lw=2.2, alpha=0.9,
                label=f'Adapt {label}')
        ax.scatter(*pt[0],         s=40, color=c, marker='o', zorder=5)
        ax.scatter(*pol['x_star'], s=80, color=c, marker='*', zorder=5)
        jp_a = pol['joints_pos']
        ax.plot(jp_a[:,0], jp_a[:,1], jp_a[:,2], '-.o', color=c,
                lw=1.0, ms=4, alpha=0.55)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('Stage 3 (PyBullet) — SE(3) EMP: Original + 3 Adaptations',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    _set_axes_equal(ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_ori_tracking(traj_orig, adapted_policies, model,
                      save_path="s3_orientation.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _, qt = traj_orig
    t  = np.arange(len(qt)) * model['dt']
    for j, lb in enumerate(['w','qx','qy','qz']):
        axes[0].plot(t, qt[:,j], lw=1.8, label=lb)
    axes[0].axhline(model['q_star'][0], ls='--', color='gray', alpha=0.5)
    axes[0].set_title('Quaternion components — original rollout', fontsize=10)
    axes[0].set_xlabel('time (s)'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    err_o = np.array([np.linalg.norm(log_quat(qt[i], model['q_star']))
                      for i in range(len(qt))])
    axes[1].plot(t, np.degrees(err_o), color=C_ORIG, lw=2, label='Original')
    for (label, (pol, traj)), c in zip(adapted_policies.items(),
                                        [C_ADAPT, C_ADAPT2, C_ADAPT3]):
        _, qt_a = traj
        t_a   = np.arange(len(qt_a)) * pol['dt']
        err_a = np.array([np.linalg.norm(log_quat(qt_a[i], pol['q_star']))
                          for i in range(len(qt_a))])
        axes[1].plot(t_a, np.degrees(err_a), color=c, lw=1.8, label=f'Adapt {label}')
    axes[1].set_title('Orientation error ‖log_{q*}(q)‖  (deg)', fontsize=10)
    axes[1].set_xlabel('time (s)'); axes[1].set_ylabel('degrees')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Stage 3 (PyBullet) — Orientation Tracking', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_summary_table(model, adapted_policies, original_traj,
                       save_path="s3_summary.png") -> None:
    rows = ['Original'] + list(adapted_policies.keys())
    cols = ['Dist to x* (cm)', 'GAS (pos)', 'GAS (ori)', 'Lyapnv viol (%)']
    data = []

    pos_o, _ = original_traj
    fd = np.linalg.norm(pos_o[-1] - model['x_star'])
    ns, nt = stability_check(model['Ak_pos'], model['P'])
    ns_o = sum(1 for A in model['Ak_ori']
               if np.linalg.eigvals(A+A.T).real.max() < 0)
    gmat = gamma_pos_batch(model['X_train'], model['means'],
                           model['covs'],   model['priors'])
    Z    = model['X_train'] - model['x_star']
    Xdp  = sum(gmat[:,k:k+1]*(Z@model['Ak_pos'][k].T)
               for k in range(model['K']))
    viol = lyapunov_violation(model['X_train'], Xdp,
                               model['P'], model['x_star'])
    data.append([f"{fd*100:.1f}", f"{ns}/{nt}", f"{ns_o}/{nt}",
                 f"{100*viol:.1f}"])

    for label, (pol, traj) in adapted_policies.items():
        pos_a, _ = traj
        fd_a = np.linalg.norm(pos_a[-1] - pol['x_star'])
        ns_a, nt_a = stability_check(pol['Ak_pos'], pol['P'])
        ns_ao = sum(1 for A in pol['Ak_ori']
                    if np.linalg.eigvals(A+A.T).real.max() < 0)
        gmat_a = gamma_pos_batch(model['X_train'], pol['means'],
                                 pol['covs'], pol['priors'])
        Z_a   = model['X_train'] - pol['x_star']          # ← per-policy attractor
        Xdp_a = sum(gmat_a[:,k:k+1] * (Z_a @ pol['Ak_pos'][k].T)
                    for k in range(pol['K']))
        viol_a = lyapunov_violation(model['X_train'], Xdp_a, pol['P'], pol['x_star'])
        data.append([f"{fd_a*100:.1f}", f"{ns_a}/{nt_a}", f"{ns_ao}/{nt_a}",
                     f"{100*viol_a:.1f}"])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.4, 2.0)
    colours_row = ['#D9EAD3', '#CFE2F3', '#FFF2CC', '#FCE5CD']
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#666')
        tbl[0, j].set_text_props(color='w')
    for i, c in enumerate(colours_row):
        for j in range(len(cols)):
            tbl[i+1, j].set_facecolor(c)
    ax.set_title('Stage 3 (PyBullet) — EMP Metrics: Original vs Adapted',
                 fontsize=12, pad=20, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(0)
    t_total = time.perf_counter()
    OUT = os.environ.get("OUT_DIR", ".")
    os.makedirs(OUT, exist_ok=True)

    # ── Load or generate demos ────────────────────────────────────────────────
    demo_path = os.path.join(OUT, "demos.npz")
    if os.path.isfile(demo_path):
        print(f"Loading existing demos from {demo_path} …")
        demos = load_demos(demo_path)
    else:
        print("Generating demonstrations (this will take a few minutes) …")
        demos, _ = generate_demonstrations(N_DEMOS=7, seed=42)
        save_demos(demos, demo_path)

    data = preprocess_demo(demos[0], N_max=150)
    print(f"  Training data: N={len(data['X'])},  K selected by BIC")
    print(f"  x* = {np.round(data['x_star'],3)}")
    print(f"  q* = {np.round(data['q_star'],4)}")

    # ── Phase 1: Train ────────────────────────────────────────────────────────
    print("\n── Phase 1: Training EMP …")
    t0    = time.perf_counter()
    model = train_emp(data)
    print(f"  K={model['K']}   training time: {time.perf_counter()-t0:.1f}s")
    ns, nt = stability_check(model['Ak_pos'], model['P'])
    print(f"  Position GAS: {ns}/{nt}")
    ns_o = sum(1 for A in model['Ak_ori']
               if np.linalg.eigvals(A+A.T).real.max() < 0)
    print(f"  Orientation A_k ≺ 0: {ns_o}/{model['K']}")

    # ── Phase 3: Original rollout ─────────────────────────────────────────────
    print("\n── Original EMP rollout …")
    traj_orig = rollout_se3(
        data['X'][0], data['Q'][0],
        model['means'], model['covs'], model['priors'],
        model['Ak_pos'], model['x_star'],
        model['mu3d'],  model['cov3d'], model['priors_o'],
        model['Ak_ori'], model['q_star'], dt=model['dt'])
    pos_o, _ = traj_orig
    print(f"  Steps: {len(pos_o)}   "
          f"final dist: {np.linalg.norm(pos_o[-1]-model['x_star'])*100:.1f}cm")

    # ── Phase 2: Three OOD Adaptations ───────────────────────────────────────
    print("\n── Phase 2: Adapting to 3 OOD scenes …")
    scenes_ood = {
        "A-translate": dict(rack_pos=np.array([0.55, 0.05, 0.15]),
                            rack_euler=np.zeros(3)),
        "B-rotate":    dict(rack_pos=np.array([0.55, -0.10, 0.15]),
                            rack_euler=np.array([0., 0., np.radians(25)])),
        "C-full-OOD":  dict(rack_pos=np.array([0.60, 0.08, 0.18]),
                            rack_euler=np.array([0., 0., np.radians(-20)])),
    }

    adapted_policies = {}
    for label, sc in scenes_ood.items():
        print(f"\n  Adapting: {label}")
        t0      = time.perf_counter()
        scene_a = BookRackScene()
        scene_a.set_rack_pose(sc['rack_pos'], sc['rack_euler'])
        new_x_star  = scene_a.get_insert_pose()[:3, 3]
        new_q_star  = pose_to_quat(scene_a.get_insert_pose())[1]
        T_start_a, _ = scene_a.get_start_pose()
        new_x_start  = T_start_a[:3, 3]
        new_q_start  = pose_to_quat(T_start_a)[1]

        pol = adapt_emp(model, new_x_start, new_x_star,
                        new_q_start, new_q_star)
        traj_a = rollout_se3(
            new_x_start, new_q_start,
            pol['means'], pol['covs'], pol['priors'],
            pol['Ak_pos'], pol['x_star'],
            pol['mu3d'],  pol['cov3d'], pol['priors_o'],
            pol['Ak_ori'], pol['q_star'], dt=pol['dt'])
        pos_a, _ = traj_a
        ns_a, nt_a = stability_check(pol['Ak_pos'], pol['P'])
        print(f"    adapt: {time.perf_counter()-t0:.1f}s  "
              f"GAS: {ns_a}/{nt_a}  "
              f"final dist: {np.linalg.norm(pos_a[-1]-pol['x_star'])*100:.1f}cm")
        adapted_policies[label] = (pol, traj_a)

    # ── Tests ─────────────────────────────────────────────────────────────────
    passed, total = run_stage3_tests(model, adapted_policies, demos, traj_orig)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n── Generating figures …")
    plot_se3_policy(model, traj_orig, adapted_policies, demos[0],
                    save_path=os.path.join(OUT, "s3_se3_policy.png"))
    plot_ori_tracking(traj_orig, adapted_policies, model,
                      save_path=os.path.join(OUT, "s3_orientation.png"))
    plot_summary_table(model, adapted_policies, traj_orig,
                       save_path=os.path.join(OUT, "s3_summary.png"))

    # ── PyBullet physical execution of original rollout ───────────────────────
    print("\n── Executing original rollout in PyBullet …")
    pos_traj_np, quat_traj_np = traj_orig
    scene_base = BookRackScene()
    pb_result  = execute_policy_in_sim(
        model,
        pos_traj_np, quat_traj_np,
        scene     = scene_base,
        save_path = os.path.join(OUT, "s3_pb_execution.png"),
    )
    final_dist_pb = float(np.linalg.norm(
        pb_result['recorded_pos'][-1] - model['x_star']))
    print(f"  PyBullet execution: {len(pb_result['recorded_pos'])} steps  "
          f"final dist: {final_dist_pb*100:.1f}cm")

    t_all = time.perf_counter() - t_total
    print(f"\nTotal wall time: {t_all:.1f}s")

    if passed == total:
        print("\n✓ Stage 3 (PyBullet) COMPLETE")
    else:
        print(f"\n✗ Stage 3: {total-passed} tests failed")
