"""
task_book_rack.py — Book-insertion scene and task definition
============================================================
Concrete SceneBase for the Franka Panda book-insertion task.

Task description
----------------
The arm STARTS already holding the book (no gripper / pick logic needed).
It performs a 3-phase insertion motion:

    Phase 1 — "approach"  : Move from start pose to the rack's slot entrance
    Phase 2 — "insert"    : Push the book forward into the slot
    Phase 3 — "retract"   : Pull the EE back slightly

The book is a kinematic rigid body that is teleported to follow the EE
at every demo step, making it visible in the PyBullet viewer.

Usage
-----
    from task_book_rack import BookInsertScene, BOOK_INSERT_TASK

    scene = BookInsertScene()
    scene.spawn_in_sim(sim)          # creates rack + book in PyBullet

    demos = generate_demonstrations(BOOK_INSERT_TASK, scene, N_demos=7, sim=sim)

    # For adaptation, vary the rack position:
    scene.set_variant(rack_pos=np.array([0.60, 0.05, 0.15]),
                      rack_euler=np.array([0, 0, np.radians(20)]))
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pybullet as pb

from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat, ik, Q_NEUTRAL
from emp_scene_base import SceneBase
from emp_task_spec import Phase, TaskSpec


# ══════════════════════════════════════════════════════════════════════════════
# BookInsertScene
# ══════════════════════════════════════════════════════════════════════════════

#: Default rack centre position (world frame, table top at z=0)
_DEFAULT_RACK_POS = np.array([0.55, -0.10, 0.15])

#: Book dimensions [thickness × width × height] in metres.
#: The thin face (thickness) is along the insertion axis.
_BOOK_SIZE = np.array([0.025, 0.18, 0.24])

#: Offset from EE origin to book centre.
#: Keep the book's thickness axis aligned with the rack insertion axis, so the
#: held book orientation matches the demonstrated insertion motion.
_T_EE_TO_BOOK_DEFAULT = make_pose(np.array([0.07, 0.0, 0.0]))


class BookInsertScene(SceneBase):
    """
    Bookrack insertion scene.

    Geometry (world frame, table top at z = 0)
    -------------------------------------------
    Robot base    : [0, 0, 0]
    Rack centre   : rack_pos   (default [0.55, -0.10, 0.15])
    Slot goal     : = rack centre  (EE inserts until EE ≡ rack centre)
    Slot entrance : 20 cm in front of rack face (along rack's −x in local frame)
    Start pose    : 10 cm further back from entrance, 5 cm higher

    Parameters
    ----------
    rack_pos       : (3,) rack centre in world frame
    rack_euler_xyz : (3,) rack rotation (Euler XYZ, radians)
    book_size      : (3,) [thickness, width, height]  (metres)
    T_ee_to_book   : (4,4) fixed transform from EE frame to book centre frame
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
            self,
            rack_pos:       np.ndarray = None,
            rack_euler_xyz: np.ndarray = None,
            book_size:      np.ndarray = None,
            T_ee_to_book:   np.ndarray = None,
    ) -> None:
        self._rack_pos_default = (
            _DEFAULT_RACK_POS.copy() if rack_pos is None else np.asarray(rack_pos, float)
        )
        rack_euler = np.zeros(3) if rack_euler_xyz is None else np.asarray(rack_euler_xyz, float)

        self.book_size    = _BOOK_SIZE.copy()    if book_size    is None else np.asarray(book_size,    float)
        self._T_ee_to_book = (_T_EE_TO_BOOK_DEFAULT.copy()
                               if T_ee_to_book is None
                               else np.asarray(T_ee_to_book, float))

        self._rack_bid: int = -1
        self._book_bid: int = -1

        self.T_rack: np.ndarray = None        # set by _rebuild_poses
        self.T_slot_world: np.ndarray = None
        self.T_approach:   np.ndarray = None
        self.T_start:      np.ndarray = None
        self.T_retract:    np.ndarray = None

        self._rebuild_poses(self._rack_pos_default, rack_euler)

    # ── Internal geometry ─────────────────────────────────────────────────────

    def _rebuild_poses(self, rack_pos: np.ndarray,
                       rack_euler: np.ndarray) -> None:
        """Recompute all derived SE(3) poses from rack position + orientation."""
        self.T_rack = make_pose(rack_pos, euler_xyz=rack_euler)

        # Slot: EE must reach the rack's centre with matching orientation
        self.T_slot_world = self.T_rack.copy()

        # Approach: 20 cm in front of the rack face
        # "In front" = along rack's local −x (approaching from outside)
        approach_offset   = np.array([-0.20, 0.0, 0.0])
        self.T_approach   = self.T_rack @ make_pose(approach_offset)
        # Orientation at approach = same as slot (book already aligned)
        self.T_approach[:3, :3] = self.T_slot_world[:3, :3]

        # Start: a bit further back and slightly above the approach pose
        # This gives the IK room to settle and gives each demo a natural
        # "ready to insert" starting configuration.
        self.T_start = self.T_approach.copy()
        self.T_start[:3, 3] += -self.T_rack[:3, 0] * 0.10   # 10 cm further from rack
        self.T_start[:3, 3] += np.array([0.0, 0.0, 0.05])   # 5 cm higher

        # Retract: pull the EE 8 cm back from the slot (post-insertion)
        self.T_retract = self.T_slot_world.copy()
        self.T_retract[:3, 3] += -self.T_rack[:3, 0] * 0.08

    # ── SceneBase interface ───────────────────────────────────────────────────

    def get_start_T(self) -> np.ndarray:
        """EE pose at the start of a demo (arm already holding the book)."""
        return self.T_start.copy()

    def get_phase_goals(self) -> Dict[str, np.ndarray]:
        """
        Named goal poses for the three insertion phases.

        Keys must match the goal_key values in BOOK_INSERT_TASK.
        """
        return {
            "approach": self.T_approach.copy(),
            "insert":   self.T_slot_world.copy(),
            "retract":  self.T_retract.copy(),
        }

    def get_goal_T(self) -> np.ndarray:
        """The EMP attractor = the fully-inserted slot pose."""
        return self.T_slot_world.copy()

    def spawn_in_sim(self, sim: PandaSimEnv) -> None:
        """
        Create PyBullet rigid bodies for the bookrack and the book.

        The rack is static (mass = 0).
        The book is kinematic (mass = 0) and starts at the EE's start pose.
        Both are removed and re-created if called again.
        """
        # ── Remove stale bodies ────────────────────────────────────────────
        for bid in (self._rack_bid, self._book_bid):
            if bid >= 0:
                try:
                    pb.removeBody(bid, physicsClientId=sim._pcid)
                    if bid in sim._scene_bodies:
                        sim._scene_bodies.remove(bid)
                except Exception:
                    pass
        self._rack_bid = -1
        self._book_bid = -1

        # ── Rack ──────────────────────────────────────────────────────────
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

        # ── Book — starts at the EE's canonical start pose ────────────────
        # Because the arm STARTS already holding the book, we place the book
        # at the position the EE will be at demo start (= T_start @ T_ee_to_book).
        T_book_init = self.T_start @ self._T_ee_to_book
        book_pos, book_q_wxyz = pose_to_quat(T_book_init)
        book_xyzw = [book_q_wxyz[1], book_q_wxyz[2],
                     book_q_wxyz[3], book_q_wxyz[0]]
        self._book_bid = sim.add_box(
            pos          = book_pos,
            half_extents = self.book_size / 2,
            orn_xyzw     = book_xyzw,
            rgba         = (0.95, 0.75, 0.25, 1.0),   # golden yellow
            mass         = 0.0,                         # kinematic — no physics
        )

    def update_held_objects(self, sim: PandaSimEnv, T_ee: np.ndarray) -> None:
        """
        Teleport the book to follow the EE.

        Called every demo step by TaskSpec.generate_demo().
        The book's pose = T_ee @ T_ee_to_book (fixed offset in EE frame).
        """
        if self._book_bid < 0:
            return
        T_book = T_ee @ self._T_ee_to_book
        pos, q_wxyz = pose_to_quat(T_book)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._book_bid, pos.tolist(), q_xyzw,
            physicsClientId=sim._pcid,
        )

    def set_variant(self,
                    rack_pos:   np.ndarray = None,
                    rack_euler: np.ndarray = None,
                    **_kwargs) -> None:
        """
        Reconfigure the rack for a new variant.

        Parameters
        ----------
        rack_pos   : new rack centre (3,)
        rack_euler : new rack rotation Euler XYZ (3,)
        """
        if rack_pos is None:
            rack_pos   = self.T_rack[:3, 3].copy()
        if rack_euler is None:
            from scipy.spatial.transform import Rotation as Rot
            rack_euler = Rot.from_matrix(self.T_rack[:3, :3]).as_euler('xyz')
        self._rebuild_poses(np.asarray(rack_pos, float),
                             np.asarray(rack_euler, float))

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_start_q(self) -> np.ndarray:
        """
        Solve IK for the canonical start pose and return joint angles.

        Cached after the first call within a given scene configuration.
        """
        q, _ok, _err = ik(self.T_start, q0=Q_NEUTRAL)
        return q

    def visualise(self, sim: PandaSimEnv) -> np.ndarray:
        """Capture and return an RGBA screenshot from the default camera."""
        return sim.capture_screenshot(640, 480)
    
    


# ══════════════════════════════════════════════════════════════════════════════
# Rack-jitter scene variant function (for generate_demonstrations)
# ══════════════════════════════════════════════════════════════════════════════

def book_rack_jitter(scene: BookInsertScene,
                     demo_idx: int,
                     rng: np.random.Generator) -> None:
    """
    Standard scene-variant function for generate_demonstrations.

    Applies a small random XY jitter to the rack position each demo.
    Z is kept fixed so the rack stays on the table.

    Usage::

        demos = generate_demonstrations(
            BOOK_INSERT_TASK, scene, N_demos=7,
            scene_variant_fn=book_rack_jitter,
        )
    """
    dp = rng.uniform(-0.012, 0.012, 3)
    dp[2] = 0.0    # keep on table
    new_pos = _DEFAULT_RACK_POS + dp
    scene.set_variant(rack_pos=new_pos)


# ══════════════════════════════════════════════════════════════════════════════
# BOOK_INSERT_TASK  —  the canonical TaskSpec
# ══════════════════════════════════════════════════════════════════════════════

BOOK_INSERT_TASK = TaskSpec(
    name="book_insert",
    phases=[
        # ─ Phase 1: move from start to slot entrance ──────────────────────
        Phase(
            name       = "approach",
            goal_key   = "approach",
            n_steps    = 40,
            jitter_pos = 0.025,   # ±2.5 cm horizontal variation per demo
            jitter_z   = 0.015,   # ±1.5 cm vertical variation
            arc_height = 0.0,     # straight line (no arc needed — arm already aligned)
        ),
        # ─ Phase 2: push into slot ────────────────────────────────────────
        Phase(
            name       = "insert",
            goal_key   = "insert",
            n_steps    = 30,
            jitter_pos = 0.008,   # small jitter to keep demos diverse
            arc_height = 0.0,
        ),
       
    ],
    dt = 0.01,   # 10 ms / step  →  ~85 steps = 0.85 s per demo
)


# ══════════════════════════════════════════════════════════════════════════════
# Variant scenes for OOD adaptation experiments
# ══════════════════════════════════════════════════════════════════════════════

def make_ood_scenes() -> dict:
    """
    Return a dict of BookInsertScene instances for three OOD configurations.

    Usage::

        ood = make_ood_scenes()
        for label, ood_scene in ood.items():
            pol = adapt_emp(model,
                            new_x_start = ood_scene.get_start_T()[:3, 3],
                            new_x_star  = ood_scene.get_goal_T()[:3, 3],
                            ...)
    """
    return {
        "translate": BookInsertScene(
            rack_pos = np.array([0.55, 0.05, 0.15]),
        ),
        "rotate": BookInsertScene(
            rack_pos       = np.array([0.55, -0.10, 0.15]),
            rack_euler_xyz = np.array([0., 0., np.radians(25)]),
        ),
        "full_ood": BookInsertScene(
            rack_pos       = np.array([0.60, 0.08, 0.18]),
            rack_euler_xyz = np.array([0., 0., np.radians(-20)]),
        ),
    }
