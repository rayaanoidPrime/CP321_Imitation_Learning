"""
pb_stage1_env.py — PyBullet Robot Arm Environment
==================================================
Drop-in replacement for stage1_env.py.
Provides the SAME public API (fk, ik, jacobian, BookRackScene, …) but
uses a Franka Panda URDF loaded in PyBullet for FK and IK, with DLS
as automatic fallback.

Design:
  • PandaSimEnv   — class that manages the PyBullet physics client
  • Module-level  — fk(), ik(), jacobian() wrappers around a lazy singleton
  • BookRackScene — same interface; get_start_pose() uses PyBullet IK
  • All pb_utils symbols are re-exported for drop-in compatibility
"""

from __future__ import annotations

import os
import sys
import atexit
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pybullet as pb
import pybullet_data

# Re-export every public symbol from pb_utils so callers can do
#   from pb_stage1_env import fk, ik, Q_NEUTRAL, BookRackScene, ...
from pb_utils import (  # noqa: F401  (public re-exports)
    DH, JOINT_LIMITS, Q_NEUTRAL,
    dh_transform, fk_dh, jacobian_dh, ik_dls,
    make_pose, pose_to_quat, se3_interpolate,
    quat_mul, quat_conjugate, quat_log, quat_exp,
    draw_arm, draw_box, draw_frame, visualise_scene, _set_axes_equal,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PandaSimEnv — PyBullet physics wrapper
# ══════════════════════════════════════════════════════════════════════════════

class PandaSimEnv:
    """
    Manages a PyBullet physics client with a Franka Panda robot.

    Parameters
    ----------
    gui     : bool  — open an interactive OpenGL window
    gravity : float — gravitational acceleration (m/s²), default -9.81

    Usage (context-manager or explicit close):
        with PandaSimEnv(gui=True) as sim:
            T_ee, T_list = sim.fk(q)
    """

    ARM_JOINT_NAMES    = [f"panda_joint{i}" for i in range(1, 8)]
    FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
    EE_LINK_NAME       = "panda_hand"

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self, gui: bool = False, gravity: float = -9.81) -> None:
        self.gui     = gui
        self._pcid   = -1          # physics-client ID
        self._rid    = -1          # robot body ID
        self._table_id = -1

        self._arm_jids:    List[int] = []   # sorted joint indices for 7 DOF
        self._ee_lid:      int       = -1   # EE link index (panda_hand)
        self._finger_jids: List[int] = []

        self._scene_bodies: List[int] = []  # extra scene objects
        self._gravity = gravity

        self._init_physics()
        self._load_robot()
        self._load_scene()
        self.set_joint_angles(Q_NEUTRAL)

    # ── private helpers ───────────────────────────────────────────────────────

    def _init_physics(self) -> None:
        try:
            mode = pb.GUI if self.gui else pb.DIRECT
            self._pcid = pb.connect(mode)
        except Exception as exc:
            raise RuntimeError(f"PyBullet connect failed: {exc}") from exc

        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._pcid)
        pb.setGravity(0, 0, self._gravity, physicsClientId=self._pcid)
        pb.setTimeStep(1.0 / 240.0, physicsClientId=self._pcid)
        pb.setRealTimeSimulation(0, physicsClientId=self._pcid)

        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_SHADOWS, 0, physicsClientId=self._pcid)
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_GUI, 0, physicsClientId=self._pcid)
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.4, cameraPitch=-30, cameraYaw=50,
                cameraTargetPosition=[0.45, 0.0, 0.25],
                physicsClientId=self._pcid)

    def _load_robot(self) -> None:
        data_path  = pybullet_data.getDataPath()
        urdf_path  = os.path.join(data_path, "franka_panda", "panda.urdf")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"Panda URDF not found at:\n  {urdf_path}\n"
                "Run:  pip install pybullet"
            )

        self._rid = pb.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=pb.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self._pcid,
        )
        self._discover_joints()

        # Set velocity-motor damping on arm joints
        for jid in self._arm_jids:
            pb.changeDynamics(
                self._rid, jid,
                linearDamping=0.0, angularDamping=0.0,
                jointDamping=0.5,
                physicsClientId=self._pcid,
            )

    def _discover_joints(self) -> None:
        """Scan URDF joints to find arm / finger / EE link indices."""
        n = pb.getNumJoints(self._rid, physicsClientId=self._pcid)
        arm_map: dict = {}

        for i in range(n):
            info       = pb.getJointInfo(self._rid, i, physicsClientId=self._pcid)
            jname      = info[1].decode()
            child_name = info[12].decode()

            if jname in self.ARM_JOINT_NAMES:
                arm_map[jname] = i

            if jname in self.FINGER_JOINT_NAMES:
                self._finger_jids.append(i)

            if child_name == self.EE_LINK_NAME:
                self._ee_lid = i   # joint index == child-link index in PyBullet

        self._arm_jids = [
            arm_map[f"panda_joint{k}"]
            for k in range(1, 8)
            if f"panda_joint{k}" in arm_map
        ]

        if len(self._arm_jids) != 7:
            raise RuntimeError(
                f"Expected 7 arm joints, found {len(self._arm_jids)}. "
                "Possibly wrong URDF version."
            )

        if self._ee_lid < 0:
            # Graceful fallback: one beyond last arm joint
            self._ee_lid = self._arm_jids[-1] + 1
            warnings.warn(
                f"EE link '{self.EE_LINK_NAME}' not found; using link {self._ee_lid}",
                stacklevel=3,
            )

    def _load_scene(self) -> None:
        """Load ground plane + table box."""
        try:
            pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.021],
                        physicsClientId=self._pcid)
        except Exception:
            pass  # plane is optional

        # Table (4 cm thick, top surface at z = 0)
        c = pb.createCollisionShape(pb.GEOM_BOX,
                                    halfExtents=[0.45, 0.4, 0.02],
                                    physicsClientId=self._pcid)
        v = pb.createVisualShape(pb.GEOM_BOX,
                                 halfExtents=[0.45, 0.4, 0.02],
                                 rgbaColor=[0.784, 0.678, 0.498, 1.0],
                                 physicsClientId=self._pcid)
        self._table_id = pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=c, baseVisualShapeIndex=v,
            basePosition=[0.4, 0.0, -0.02],
            physicsClientId=self._pcid,
        )

    # ── kinematics ────────────────────────────────────────────────────────────

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward kinematics via PyBullet getLinkState.

        Returns
        -------
        T_ee    : (4,4) EE (panda_hand) transform
        T_list  : list of 8 transforms  [base, link1 … link7]
                  matches stage1_env.fk() output exactly
        """
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (7,):
            raise ValueError(f"q must be shape (7,), got {q.shape}")

        # Set joint angles (hard reset — no physics propagation needed for FK)
        for jid, qval in zip(self._arm_jids, q):
            pb.resetJointState(self._rid, jid, float(qval),
                               physicsClientId=self._pcid)

        # Base transform
        bpos, born = pb.getBasePositionAndOrientation(
            self._rid, physicsClientId=self._pcid)
        T_list = [_xyzw_to_T(bpos, born)]

        # Arm link transforms (children of arm joints)
        for jid in self._arm_jids:
            state = pb.getLinkState(
                self._rid, jid,
                computeForwardKinematics=True,
                physicsClientId=self._pcid,
            )
            # state[4] = worldLinkFramePosition (after FK)
            # state[5] = worldLinkFrameOrientation xyzw
            T_list.append(_xyzw_to_T(state[4], state[5]))

        # EE (panda_hand)
        ee = pb.getLinkState(
            self._rid, self._ee_lid,
            computeForwardKinematics=True,
            physicsClientId=self._pcid,
        )
        T_ee = _xyzw_to_T(ee[4], ee[5])

        return T_ee, T_list

    def ik(self,
           T_des: np.ndarray,
           q0:    np.ndarray = None,
           n_iter: int       = 300,
           tol:   float      = 1e-3) -> Tuple[np.ndarray, bool, float]:
        """
        IK via PyBullet calculateInverseKinematics, with DLS fallback.

        Returns (q (7,), success: bool, pos_err: float)
        """
        if q0 is None:
            q0 = Q_NEUTRAL.copy()
        q0 = np.clip(q0, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        target_pos = T_des[:3, 3].tolist()
        _, tq_wxyz  = pose_to_quat(T_des)
        # PyBullet quaternion convention: xyzw
        tq_xyzw = [float(tq_wxyz[1]), float(tq_wxyz[2]),
                   float(tq_wxyz[3]), float(tq_wxyz[0])]

        lo  = JOINT_LIMITS[:, 0].tolist()
        hi  = JOINT_LIMITS[:, 1].tolist()
        rng = (JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]).tolist()

        best_q   = q0.copy()
        best_err = np.inf

        rest_poses_list = [
            q0.tolist(),
            Q_NEUTRAL.tolist(),
            (JOINT_LIMITS[:, 0] * 0.3 + JOINT_LIMITS[:, 1] * 0.7).tolist(),
        ]

        for rest in rest_poses_list:
            # Warm-start: set robot to rest pose before calling IK
            for jid, qval in zip(self._arm_jids, rest):
                pb.resetJointState(self._rid, jid, float(qval),
                                   physicsClientId=self._pcid)

            raw = pb.calculateInverseKinematics(
                self._rid,
                self._ee_lid,
                targetPosition=target_pos,
                targetOrientation=tq_xyzw,
                lowerLimits=lo,
                upperLimits=hi,
                jointRanges=rng,
                restPoses=rest,
                maxNumIterations=n_iter,
                residualThreshold=tol * 0.5,
                physicsClientId=self._pcid,
            )

            q_candidate = np.clip(
                np.array(raw[:7], dtype=float),
                JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
            )
            T_fk, _ = self.fk(q_candidate)
            err = float(np.linalg.norm(T_fk[:3, 3] - T_des[:3, 3]))

            if err < best_err:
                best_err = err
                best_q   = q_candidate.copy()

            if best_err < tol:
                break

        # ── DLS fallback ────────────────────────────────────────────────────
        if best_err > tol * 3.0:
            q_dls, ok_dls, err_dls = ik_dls(T_des, q0, n_iter=300, tol=tol)
            q_dls = np.clip(np.asarray(q_dls, dtype=float),
                            JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            T_dls_pb, _ = self.fk(q_dls)
            err_dls_pb = float(np.linalg.norm(T_dls_pb[:3, 3] - T_des[:3, 3]))
            if err_dls_pb < best_err:
                best_err = err_dls_pb
                best_q   = q_dls

        return best_q, bool(best_err < tol * 5.0), float(best_err)

    def jacobian(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Numerical 6×7 Jacobian via central FD using PyBullet FK."""
        q = np.asarray(q, dtype=float)
        J = np.zeros((6, 7))
        for i in range(7):
            qp, qm = q.copy(), q.copy()
            qp[i] += delta
            qm[i] -= delta
            Tp, _ = self.fk(qp)
            Tm, _ = self.fk(qm)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * delta)
            dR = Tp[:3, :3] @ Tm[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            if abs(angle) < 1e-10:
                J[3:, i] = 0.0
            else:
                skew = (dR - dR.T) / (2 * np.sin(angle))
                J[3:, i] = np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) \
                            * angle / (2 * delta)
        return J

    # ── joint control ─────────────────────────────────────────────────────────

    def set_joint_angles(self, q: np.ndarray, control: bool = False) -> None:
        """Set arm joint angles.  control=True uses PD; False does hard reset."""
        q = np.clip(np.asarray(q, dtype=float),
                    JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        for jid, qval in zip(self._arm_jids, q):
            if control:
                pb.setJointMotorControl2(
                    self._rid, jid,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=float(qval),
                    force=200.0,
                    physicsClientId=self._pcid,
                )
            else:
                pb.resetJointState(self._rid, jid, float(qval),
                                   physicsClientId=self._pcid)

        # Keep fingers closed (gripper = 0 m)
        for fid in self._finger_jids:
            pb.resetJointState(self._rid, fid, 0.0,
                               physicsClientId=self._pcid)

    def open_gripper(self, width: float = 0.08) -> None:
        """Open the parallel gripper to the given width (m)."""
        half = width / 2.0
        for fid in self._finger_jids:
            pb.resetJointState(self._rid, fid, float(half),
                               physicsClientId=self._pcid)

    def get_joint_angles(self) -> np.ndarray:
        return np.array([
            pb.getJointState(self._rid, jid, physicsClientId=self._pcid)[0]
            for jid in self._arm_jids
        ])

    # ── simulation ────────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            pb.stepSimulation(physicsClientId=self._pcid)

    def reset(self) -> None:
        self.set_joint_angles(Q_NEUTRAL)

    # ── scene helpers ─────────────────────────────────────────────────────────

    def add_box(self,
                pos:        np.ndarray,
                half_extents: np.ndarray,
                orn_xyzw:   np.ndarray = None,
                rgba:       tuple      = (0.6, 0.4, 0.2, 0.85),
                mass:       float      = 0.0) -> int:
        """Add a box body to the simulation. Returns body ID."""
        if orn_xyzw is None:
            orn_xyzw = [0, 0, 0, 1]
        c = pb.createCollisionShape(pb.GEOM_BOX,
                                    halfExtents=half_extents.tolist(),
                                    physicsClientId=self._pcid)
        v = pb.createVisualShape(pb.GEOM_BOX,
                                 halfExtents=half_extents.tolist(),
                                 rgbaColor=list(rgba),
                                 physicsClientId=self._pcid)
        bid = pb.createMultiBody(
            baseMass=float(mass),
            baseCollisionShapeIndex=c,
            baseVisualShapeIndex=v,
            basePosition=pos.tolist(),
            baseOrientation=list(orn_xyzw),
            physicsClientId=self._pcid,
        )
        self._scene_bodies.append(bid)
        return bid

    def remove_scene_bodies(self) -> None:
        """Remove all previously added scene bodies."""
        for bid in self._scene_bodies:
            try:
                pb.removeBody(bid, physicsClientId=self._pcid)
            except Exception:
                pass
        self._scene_bodies.clear()

    # ── rendering ────────────────────────────────────────────────────────────

    def capture_screenshot(self,
                           width:  int = 640,
                           height: int = 480) -> np.ndarray:
        """
        Capture RGBA image from a fixed perspective camera.
        Returns ndarray (H, W, 4) uint8.
        """
        view = pb.computeViewMatrix(
            cameraEyePosition=[1.0, -0.5, 0.8],
            cameraTargetPosition=[0.4, 0.0, 0.2],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._pcid,
        )
        proj = pb.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.05, farVal=10.0,
            physicsClientId=self._pcid,
        )
        _, _, rgba, _, _ = pb.getCameraImage(
            width=width, height=height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self._pcid,
        )
        return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._pcid >= 0:
            try:
                pb.disconnect(self._pcid)
            except Exception:
                pass
            self._pcid = -1

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── property ─────────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._pcid >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Helper: (pos, xyzw) → 4×4 SE(3)
# ──────────────────────────────────────────────────────────────────────────────

def _xyzw_to_T(pos, orn_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    x, y, z, w = orn_xyzw
    T[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    return T


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Module-level singleton API
#     Provides fk(), ik(), jacobian() matching stage1_env.py's signatures
# ══════════════════════════════════════════════════════════════════════════════

_SIM:      Optional[PandaSimEnv] = None
_GUI_MODE: bool                  = False


_ATEXIT_REGISTERED = False


def _get_sim() -> PandaSimEnv:
    global _SIM, _ATEXIT_REGISTERED
    if _SIM is None or not _SIM.is_alive:
        _SIM = PandaSimEnv(gui=_GUI_MODE)
        if not _ATEXIT_REGISTERED:
            atexit.register(lambda: _SIM.close() if _SIM else None)
            _ATEXIT_REGISTERED = True
    return _SIM


def use_gui(flag: bool = True) -> None:
    """
    Call before any fk/ik calls to enable the PyBullet GUI window.
    Has no effect after the singleton has been created.
    """
    global _GUI_MODE
    _GUI_MODE = flag


def close_sim() -> None:
    """Explicitly close the singleton physics client."""
    global _SIM
    if _SIM is not None:
        _SIM.close()
        _SIM = None


# ── Public FK / IK / Jacobian ─────────────────────────────────────────────────

def fk(q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Forward kinematics (PyBullet). Same API as stage1_env.fk()."""
    return _get_sim().fk(np.asarray(q, dtype=float))


def ik(T_des:  np.ndarray,
       q0:     np.ndarray = None,
       n_iter: int        = 300,
       lam:    float      = 0.05,   # kept for API compatibility
       tol:    float      = 1e-3,
       vel_tol: float     = 1e-6    # kept for API compatibility
       ) -> Tuple[np.ndarray, bool, float]:
    """Inverse kinematics (PyBullet + DLS fallback). API ≡ stage1_env.ik()."""
    return _get_sim().ik(T_des, q0=q0, n_iter=n_iter, tol=tol)


def jacobian(q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian (6×7) using PyBullet FK."""
    return _get_sim().jacobian(np.asarray(q, dtype=float), delta=delta)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BookRackScene — same interface as stage1_env.BookRackScene
#     Uses pb ik() for get_start_pose().
# ══════════════════════════════════════════════════════════════════════════════

class BookRackScene:
    """
    Bookrack scene (same API as stage1_env.BookRackScene).
    Now also creates PyBullet rigid bodies for the rack and book.

    The robot base is at the world origin.  Table top at z = 0.
    """

    def __init__(self,
                 rack_pos:       np.ndarray = None,
                 rack_euler_xyz: np.ndarray = None,
                 book_thickness: float      = 0.03) -> None:
        if rack_pos is None:
            rack_pos = np.array([0.55, -0.10, 0.15])
        if rack_euler_xyz is None:
            rack_euler_xyz = np.zeros(3)

        self.T_rack   = make_pose(rack_pos, euler_xyz=rack_euler_xyz)
        self.T_base   = np.eye(4)
        self.book_size       = np.array([0.025, 0.18, 0.24])
        self.book_thickness  = book_thickness

        # Slot: centred inside rack, approach from -x of rack frame
        slot_offset     = np.zeros(3)
        self.T_slot_rack  = make_pose(slot_offset)
        self.T_slot_world = self.T_rack @ self.T_slot_rack
        approach_offset   = np.array([-0.20, 0.0, 0.0])
        self.T_approach   = self.T_rack @ make_pose(approach_offset)

        # PyBullet scene bodies (created lazily on first spawn call)
        self._rack_bid: int = -1
        self._book_bid: int = -1

    # ── PyBullet scene spawning ───────────────────────────────────────────────

    def spawn_in_sim(self, sim: PandaSimEnv) -> None:
        """
        Create PyBullet rigid bodies for rack and book.
        Safe to call multiple times (removes old bodies first).
        """
        # Remove old bodies if they exist
        for bid in (self._rack_bid, self._book_bid):
            if bid >= 0:
                try:
                    pb.removeBody(bid, physicsClientId=sim._pcid)
                    if bid in sim._scene_bodies:
                        sim._scene_bodies.remove(bid)
                except Exception:
                    pass

        # Bookrack (half-extents = 0.11 × 0.16 × 0.16 m)
        rack_pos, rack_q_wxyz = pose_to_quat(self.T_rack)
        rack_xyzw = [rack_q_wxyz[1], rack_q_wxyz[2],
                     rack_q_wxyz[3], rack_q_wxyz[0]]
        self._rack_bid = sim.add_box(
            pos          = rack_pos,
            half_extents = np.array([0.11, 0.16, 0.16]),
            orn_xyzw     = rack_xyzw,
            rgba         = (0.545, 0.415, 0.082, 0.85),
            mass         = 0.0,
        )

        # Book (thin flat box, starts at approach pose)
        book_pos, book_q_wxyz = pose_to_quat(self.T_approach)
        book_xyzw = [book_q_wxyz[1], book_q_wxyz[2],
                     book_q_wxyz[3], book_q_wxyz[0]]
        self._book_bid = sim.add_box(
            pos          = book_pos,
            half_extents = self.book_size / 2,
            orn_xyzw     = book_xyzw,
            rgba         = (0.9, 0.7, 0.3, 1.0),
            mass         = 0.1,   # 100 g
        )

    def move_book_to(self, sim: PandaSimEnv, T: np.ndarray) -> None:
        """Teleport book to pose T (for recording / debugging)."""
        if self._book_bid < 0:
            return
        pos, q_wxyz = pose_to_quat(T)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._book_bid, pos.tolist(), q_xyzw,
            physicsClientId=sim._pcid,
        )

    # ── Pose queries (same API as stage1_env.BookRackScene) ──────────────────

    def get_start_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Starting EE pose: 10 cm above approach pose.
        Returns (T_start (4,4), q_start (7,)).
        """
        T_start = self.T_approach.copy()
        T_start[:3, 3] += np.array([0.0, 0.0, 0.10])
        q0, _ok, _err = ik(T_start, q0=Q_NEUTRAL)
        return T_start, q0

    def get_insert_pose(self) -> np.ndarray:
        """Final EE pose: book fully inserted into slot. Returns T (4,4)."""
        return self.T_slot_world.copy()

    def set_rack_pose(self,
                      rack_pos:       np.ndarray,
                      rack_euler_xyz: np.ndarray = None) -> None:
        """Update rack pose (for adaptation experiments)."""
        if rack_euler_xyz is None:
            rack_euler_xyz = np.zeros(3)
        self.T_rack       = make_pose(rack_pos, euler_xyz=rack_euler_xyz)
        self.T_slot_world = self.T_rack @ self.T_slot_rack
        approach_offset   = np.array([-0.20, 0.0, 0.0])
        self.T_approach   = self.T_rack @ make_pose(approach_offset)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Stage 1 Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1_tests() -> Tuple[int, int, BookRackScene, np.ndarray, np.ndarray]:
    print("=" * 60)
    print("STAGE 1 TESTS — PyBullet Robot Arm Environment")
    print("=" * 60)
    passed = 0; total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        sym = "✓" if cond else "✗"
        print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))
        if cond:
            passed += 1
        return cond

    # T1: FK at neutral pose
    T, T_list = fk(Q_NEUTRAL)
    ee_pos = T[:3, 3]
    check("FK produces valid SE(3)",
          T.shape == (4, 4) and abs(float(np.linalg.det(T[:3, :3])) - 1) < 1e-5,
          f"det(R)={np.linalg.det(T[:3,:3]):.6f}")
    check("FK EE above table (z > 0)",
          float(ee_pos[2]) > 0.0,
          f"EE={np.round(ee_pos, 3)}")
    check("FK returns 8 link transforms",
          len(T_list) == 8, f"got {len(T_list)}")

    # T2: FK → IK round-trip
    np.random.seed(1)
    q_rand = np.clip(Q_NEUTRAL + np.random.randn(7) * 0.25,
                     JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    T_rand, _ = fk(q_rand)
    q_ik, ok, err = ik(T_rand, q0=Q_NEUTRAL)
    T_ik, _  = fk(q_ik)
    pos_err  = float(np.linalg.norm(T_rand[:3, 3] - T_ik[:3, 3]))
    check("IK converges to correct position",
          pos_err < 0.02, f"pos_err={pos_err*100:.2f} cm")

    # T3: Jacobian shape
    J = jacobian(Q_NEUTRAL)
    check("Jacobian shape is (6,7)", J.shape == (6, 7))
    check("Jacobian linear rows sane",
          bool(np.all(np.abs(J[:3, :]) < 2.5)),
          f"max={np.abs(J[:3,:]).max():.3f}")

    # T4: Quaternion round-trip
    from scipy.spatial.transform import Rotation as Rot
    q2 = Rot.from_euler('z', np.pi / 4).as_quat()
    q2_wxyz = np.array([q2[3], q2[0], q2[1], q2[2]])
    log_q2  = quat_log(q2_wxyz)
    q2_back = quat_exp(log_q2)
    err_q   = float(np.linalg.norm(q2_wxyz - q2_back))
    check("quat log/exp round-trip", err_q < 1e-6, f"err={err_q:.2e}")

    # T5: SE(3) interpolation midpoint
    T0 = make_pose(np.array([0.3, 0., 0.4]), euler_xyz=[0, 0, 0])
    T1 = make_pose(np.array([0.6, 0.2, 0.3]), euler_xyz=[0, 0.5, 0.3])
    Tm = se3_interpolate(T0, T1, 0.5)
    p_mid = Tm[:3, 3]
    check("SE(3) interpolation midpoint",
          np.allclose(p_mid, 0.5*(T0[:3, 3]+T1[:3, 3]), atol=1e-5),
          f"p_mid={np.round(p_mid,3)}")

    # T6: Scene construction
    scene = BookRackScene()
    T_start, q_start = scene.get_start_pose()
    T_ins = scene.get_insert_pose()
    check("Scene: start pose reachable",
          q_start is not None,
          f"start={np.round(T_start[:3,3],3)}")
    check("Scene: rack/slot are valid SE(3)",
          abs(float(np.linalg.det(scene.T_rack[:3, :3])) - 1) < 1e-5 and
          abs(float(np.linalg.det(T_ins[:3, :3]))        - 1) < 1e-5)

    print(f"\n  Result: {passed}/{total} tests passed")
    return passed, total, scene, T_start, q_start


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt

    np.random.seed(0)
    OUT = os.environ.get("OUT_DIR", ".")
    os.makedirs(OUT, exist_ok=True)

    passed, total, scene, T_start, q_start = run_stage1_tests()

    # Matplotlib scene overview
    fig = visualise_scene(scene, q=Q_NEUTRAL,
                          title="Stage 1 (PyBullet) — Robot Arm + Bookrack Scene")
    fig.savefig(os.path.join(OUT, "s1_scene.png"), dpi=150, bbox_inches='tight')
    print("Saved: s1_scene.png")

    # PyBullet screenshot
    sim = _get_sim()
    scene.spawn_in_sim(sim)
    img = sim.capture_screenshot(960, 720)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt2
    fig2, ax = plt2.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Stage 1 — PyBullet renderer view", fontsize=11)
    fig2.savefig(os.path.join(OUT, "s1_pybullet_view.png"), dpi=150,
                 bbox_inches='tight')
    print("Saved: s1_pybullet_view.png")

    if passed == total:
        print("\n✓ Stage 1 (PyBullet) COMPLETE")
    else:
        print(f"\n✗ Stage 1: {total-passed} tests failed")
