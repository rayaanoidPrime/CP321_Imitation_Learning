"""
pb_utils.py — Pure-math utilities for the PyBullet EMP pipeline
================================================================
No PyBullet dependency here.  Contains:
  • DH parameters and joint limits (Franka Panda)
  • numpy-based FK / Jacobian / DLS-IK  (kept as fallback)
  • SE(3) / quaternion math
  • Matplotlib 3-D visualization helpers
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Franka Panda DH parameters (modified DH, a/d in metres)
# ──────────────────────────────────────────────────────────────────────────────
DH = np.array([
    [0.000,  0.333,  0.000,   0.0],
    [0.000,  0.000, -np.pi/2, 0.0],
    [0.000,  0.316,  np.pi/2, 0.0],
    [0.0825, 0.000,  np.pi/2, 0.0],
    [-0.0825,0.384, -np.pi/2, 0.0],
    [0.000,  0.000,  np.pi/2, 0.0],
    [0.088,  0.107,  np.pi/2, 0.0],
])

JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

Q_NEUTRAL = np.array([-1.57, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ──────────────────────────────────────────────────────────────────────────────
# DH-based numpy FK (fallback; PyBullet FK is used in normal operation)
# ──────────────────────────────────────────────────────────────────────────────

def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Homogeneous transform for one DH link. Returns (4,4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])


def fk_dh(q: np.ndarray):
    """
    Numpy DH forward kinematics.
    Returns (T_ee (4,4),  T_list: list of 8 transforms).
    """
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    T_list = [T.copy()]
    for i in range(7):
        a, d, alpha, offset = DH[i]
        Ti = dh_transform(a, d, alpha, q[i] + offset)
        T = T @ Ti
        T_list.append(T.copy())
    return T, T_list


def jacobian_dh(q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Numerical 6×7 Jacobian via central finite differences."""
    T0, _ = fk_dh(q)
    J = np.zeros((6, 7))
    for i in range(7):
        qp, qm = q.copy(), q.copy()
        qp[i] += delta
        qm[i] -= delta
        Tp, _ = fk_dh(qp)
        Tm, _ = fk_dh(qm)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * delta)
        dR = Tp[:3, :3] @ Tm[:3, :3].T
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if abs(angle) < 1e-10:
            J[3:, i] = 0.0
        else:
            skew = (dR - dR.T) / (2 * np.sin(angle))
            J[3:, i] = np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) * angle / (2 * delta)
    return J


def ik_dls(T_des: np.ndarray, q0: np.ndarray = None,
           n_iter: int = 200, lam: float = 0.05,
           tol: float = 1e-4) -> tuple:
    """
    Damped-least-squares IK (numpy, no PyBullet).
    Returns (q, success, final_err).
    """
    q = Q_NEUTRAL.copy() if q0 is None else np.asarray(q0, dtype=float).copy()
    p_des = T_des[:3, 3]
    R_des = T_des[:3, :3]

    for _ in range(n_iter):
        T_cur, _ = fk_dh(q)
        p_err = p_des - T_cur[:3, 3]
        R_err_mat = R_des @ T_cur[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_err_mat) - 1) / 2, -1, 1))
        if abs(angle) < 1e-10:
            r_err = np.zeros(3)
        else:
            skew = (R_err_mat - R_err_mat.T) / (2 * np.sin(angle))
            axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            r_err = axis * angle

        err = np.concatenate([p_err, r_err])
        if np.linalg.norm(err) < tol:
            return q, True, np.linalg.norm(err)

        J = jacobian_dh(q)
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err)
        step = min(1.0, 0.1 / (np.linalg.norm(dq) + 1e-9))
        q = np.clip(q + step * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T_cur, _ = fk_dh(q)
    enorm = np.linalg.norm(p_des - T_cur[:3, 3])
    return q, enorm < 0.01, enorm


# ──────────────────────────────────────────────────────────────────────────────
# SE(3) / quaternion utilities
# ──────────────────────────────────────────────────────────────────────────────

def make_pose(pos: np.ndarray,
              rotvec=None,
              euler_xyz=None,
              quat_wxyz=None) -> np.ndarray:
    """Build a 4×4 homogeneous transform. Returns T."""
    T = np.eye(4)
    T[:3, 3] = pos
    if rotvec is not None:
        T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    elif euler_xyz is not None:
        T[:3, :3] = R.from_euler('xyz', euler_xyz).as_matrix()
    elif quat_wxyz is not None:
        q = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T[:3, :3] = R.from_quat(q).as_matrix()
    return T


def pose_to_quat(T: np.ndarray):
    """Extract (pos (3,), quat_wxyz (4,)) from SE(3)."""
    pos = T[:3, 3]
    qxyz = R.from_matrix(T[:3, :3]).as_quat()  # scalar-last xyzw
    return pos, np.array([qxyz[3], qxyz[0], qxyz[1], qxyz[2]])


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    """SLERP/LERP SE(3) interpolation. t ∈ [0,1]."""
    p0, q0 = pose_to_quat(T0)
    p1, q1 = pose_to_quat(T1)
    p_t = (1 - t) * p0 + t * p1
    dot = np.clip(np.dot(q0, q1), -1, 1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        qt = q0 + t * (q1 - q0)
        qt /= np.linalg.norm(qt)
    else:
        theta0 = np.arccos(dot)
        theta  = theta0 * t
        qt = (np.sin(theta0 - theta) * q0 + np.sin(theta) * q1) / np.sin(theta0)
    return make_pose(p_t, quat_wxyz=qt)


def quat_mul(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 (both wxyz)."""
    w1, x1, y1, z1 = q1_wxyz
    w2, x2, y2, z2 = q2_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])


def quat_log(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion logarithmic map → ℝ³."""
    q = np.asarray(q_wxyz, dtype=float)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-10:
        return np.zeros(3)
    theta = np.arccos(w)
    return (theta / nv) * v


def quat_exp(omega: np.ndarray) -> np.ndarray:
    """Quaternion exponential map ℝ³ → unit quaternion (wxyz)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(theta),
                     *(np.sin(theta) / theta * omega)])


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib 3-D visualization helpers (backend-agnostic, writes to files)
# ──────────────────────────────────────────────────────────────────────────────

def draw_frame(ax, T: np.ndarray, scale: float = 0.05,
               label: str = None, alpha: float = 0.9) -> None:
    """Draw xyz axes of an SE(3) frame."""
    o  = T[:3, 3]
    ex = T[:3, 0] * scale
    ey = T[:3, 1] * scale
    ez = T[:3, 2] * scale
    ax.quiver(*o, *ex, color='r', alpha=alpha, linewidth=1.5)
    ax.quiver(*o, *ey, color='g', alpha=alpha, linewidth=1.5)
    ax.quiver(*o, *ez, color='b', alpha=alpha, linewidth=1.5)
    if label:
        ax.text(*(o + ez * 1.5), label, fontsize=7, color='k')


def draw_box(ax, T: np.ndarray, size: np.ndarray,
             color: str = 'tan', alpha: float = 0.25) -> None:
    """Draw a box centred at T's origin with half-sizes size/2."""
    dx, dy, dz = size / 2
    corners = np.array([
        [-dx,-dy,-dz],[dx,-dy,-dz],[dx,dy,-dz],[-dx,dy,-dz],
        [-dx,-dy, dz],[dx,-dy, dz],[dx,dy, dz],[-dx,dy, dz],
    ])
    corners_w = (T[:3, :3] @ corners.T).T + T[:3, 3]
    faces = [
        [corners_w[j] for j in [0,1,2,3]],
        [corners_w[j] for j in [4,5,6,7]],
        [corners_w[j] for j in [0,1,5,4]],
        [corners_w[j] for j in [2,3,7,6]],
        [corners_w[j] for j in [1,2,6,5]],
        [corners_w[j] for j in [0,3,7,4]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha,
                            facecolor=color, edgecolor='gray', linewidth=0.4)
    ax.add_collection3d(poly)


def draw_arm(ax, T_list: list, color: str = '#3B8BD4', lw: float = 2.5) -> None:
    """Draw robot links as line segments using FK transform list."""
    pts = np.array([T[:3, 3] for T in T_list])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            '-o', color=color, lw=lw, ms=4, zorder=5)


def _set_axes_equal(ax) -> None:
    """Make 3-D axes have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centre = limits.mean(axis=1)
    radius = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(centre[0]-radius, centre[0]+radius)
    ax.set_ylim3d(centre[1]-radius, centre[1]+radius)
    ax.set_zlim3d(centre[2]-radius, centre[2]+radius)


def visualise_scene(scene, q: np.ndarray = None,
                    traj: list = None, title: str = "Scene") -> plt.Figure:
    """Render scene with optional arm config and trajectory (matplotlib)."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    T_table = make_pose(np.array([0.4, 0.0, -0.02]))
    draw_box(ax, T_table, np.array([0.9, 0.8, 0.04]), color='#C8AD7F', alpha=0.35)
    draw_box(ax, np.eye(4), np.array([0.10, 0.10, 0.02]), color='#444', alpha=0.7)
    draw_box(ax, scene.T_rack, np.array([0.22, 0.32, 0.32]), color='#8B6914', alpha=0.20)
    draw_frame(ax, scene.T_rack, scale=0.06, label='rack')
    draw_frame(ax, scene.T_slot_world, scale=0.05, label='slot')
    draw_frame(ax, scene.T_approach, scale=0.05, label='approach')

    if q is not None:
        _, T_list = fk_dh(q)
        draw_arm(ax, T_list)
        draw_frame(ax, T_list[-1], scale=0.07, label='EE', alpha=1.0)

    if traj is not None:
        pts = np.array([T[:3, 3] for T in traj])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                '-', color='#D85A30', lw=2, alpha=0.8, label='trajectory')
        draw_frame(ax, traj[0],  scale=0.04)
        draw_frame(ax, traj[-1], scale=0.04)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12)
    _set_axes_equal(ax)
    plt.tight_layout()
    return fig
