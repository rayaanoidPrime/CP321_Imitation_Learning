"""
task_obstacle_avoidance.py — Obstacle avoidance scene and task definition
==========================================================================
Concrete SceneBase for the Franka Panda obstacle avoidance task.

Task description
----------------
The arm performs a book-insertion-like motion while avoiding a static obstacle
(e.g. a mustard bottle) placed between the start and goal.

The obstacle avoidance is achieved through EMP's modulation mechanism:
- The nominal EMP policy generates a trajectory toward the goal
- A repulsive potential field from the obstacle modulates the velocity
- The robot circumvents the obstacle while still converging to the goal

    Phase 1 — "approach"  : Move toward the rack, navigating around the obstacle
    Phase 2 — "insert"    : Push the book into the slot

The obstacle is modelled as a sphere for the repulsive field computation.

Usage
-----
    from task_obstacle_avoidance import ObstacleAvoidanceScene, OBSTACLE_AVOIDANCE_TASK

    scene = ObstacleAvoidanceScene(obstacle_pos=np.array([0.50, -0.05, 0.15]))
    scene.spawn_in_sim(sim)

    demos = generate_demonstrations(OBSTACLE_AVOIDANCE_TASK, scene, N_demos=7, sim=sim)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pybullet as pb

from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat, ik, Q_NEUTRAL
from emp_scene_base import SceneBase
from emp_task_spec import Phase, TaskSpec


# ══════════════════════════════════════════════════════════════════════════════
# ObstacleAvoidanceScene
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_RACK_POS = np.array([0.55, -0.10, 0.15])
_DEFAULT_OBSTACLE_POS = np.array([0.50, -0.05, 0.18])

_BOOK_SIZE = np.array([0.025, 0.18, 0.24])
_T_EE_TO_BOOK_DEFAULT = make_pose(np.array([0.07, 0.0, 0.0]))

_OBSTACLE_RADIUS = 0.03  # metres, for repulsive field


class ObstacleAvoidanceScene(SceneBase):
    """
    Book-insertion with obstacle avoidance scene.

    Geometry (world frame, table top at z = 0)
    -------------------------------------------
    Robot base   : [0, 0, 0]
    Rack centre  : rack_pos   (default [0.55, -0.10, 0.15])
    Obstacle     : obstacle_pos (default [0.50, -0.05, 0.18])
    Slot goal    : = rack centre
    Slot entrance: 20 cm in front of rack face
    Start pose   : further back from entrance

    Parameters
    ----------
    rack_pos        : (3,) rack centre in world frame
    rack_euler_xyz  : (3,) rack rotation (Euler XYZ, radians)
    obstacle_pos    : (3,) obstacle centre in world frame
    obstacle_radius : float obstacle radius for repulsive field
    book_size       : (3,) book dimensions
    T_ee_to_book    : (4,4) fixed transform from EE to book centre
    """

    def __init__(
            self,
            rack_pos:       np.ndarray = None,
            rack_euler_xyz: np.ndarray = None,
            obstacle_pos:   np.ndarray = None,
            obstacle_radius: float = _OBSTACLE_RADIUS,
            book_size:      np.ndarray = None,
            T_ee_to_book:   np.ndarray = None,
    ) -> None:
        self._rack_pos_default = (
            _DEFAULT_RACK_POS.copy() if rack_pos is None else np.asarray(rack_pos, float)
        )
        rack_euler = np.zeros(3) if rack_euler_xyz is None else np.asarray(rack_euler_xyz, float)
        self._obstacle_pos = (
            _DEFAULT_OBSTACLE_POS.copy() if obstacle_pos is None else np.asarray(obstacle_pos, float)
        )
        self.obstacle_radius = obstacle_radius

        self.book_size = _BOOK_SIZE.copy() if book_size is None else np.asarray(book_size, float)
        self._T_ee_to_book = (
            _T_EE_TO_BOOK_DEFAULT.copy()
            if T_ee_to_book is None
            else np.asarray(T_ee_to_book, float)
        )

        self._rack_bid: int = -1
        self._book_bid: int = -1
        self._obstacle_bid: int = -1

        self.T_rack: np.ndarray = None
        self.T_slot_world: np.ndarray = None
        self.T_approach: np.ndarray = None
        self.T_start: np.ndarray = None

        self._rebuild_poses(self._rack_pos_default, rack_euler)

    def _rebuild_poses(self, rack_pos: np.ndarray, rack_euler: np.ndarray) -> None:
        self.T_rack = make_pose(rack_pos, euler_xyz=rack_euler)
        self.T_slot_world = self.T_rack.copy()

        approach_offset = np.array([-0.20, 0.0, 0.0])
        self.T_approach = self.T_rack @ make_pose(approach_offset)
        self.T_approach[:3, :3] = self.T_slot_world[:3, :3]

        self.T_start = self.T_approach.copy()
        self.T_start[:3, 3] += -self.T_rack[:3, 0] * 0.15
        self.T_start[:3, 3] += np.array([0.0, 0.0, 0.08])

    def get_start_T(self) -> np.ndarray:
        return self.T_start.copy()

    def get_phase_goals(self) -> Dict[str, np.ndarray]:
        return {
            "approach": self.T_approach.copy(),
            "insert":   self.T_slot_world.copy(),
        }

    def get_goal_T(self) -> np.ndarray:
        return self.T_slot_world.copy()

    def get_obstacle_pos(self) -> np.ndarray:
        return self._obstacle_pos.copy()

    def get_obstacle_radius(self) -> float:
        return self.obstacle_radius

    def spawn_in_sim(self, sim: PandaSimEnv) -> None:
        for bid in (self._rack_bid, self._book_bid, self._obstacle_bid):
            if bid >= 0:
                try:
                    pb.removeBody(bid, physicsClientId=sim._pcid)
                    if bid in sim._scene_bodies:
                        sim._scene_bodies.remove(bid)
                except Exception:
                    pass
        self._rack_bid = -1
        self._book_bid = -1
        self._obstacle_bid = -1

        # Rack
        rack_pos, rack_q_wxyz = pose_to_quat(self.T_rack)
        rack_xyzw = [rack_q_wxyz[1], rack_q_wxyz[2], rack_q_wxyz[3], rack_q_wxyz[0]]
        self._rack_bid = sim.add_box(
            pos=rack_pos,
            half_extents=np.array([0.11, 0.16, 0.16]),
            orn_xyzw=rack_xyzw,
            rgba=(0.545, 0.415, 0.082, 0.85),
            mass=0.0,
        )

        # Book
        T_book_init = self.T_start @ self._T_ee_to_book
        book_pos, book_q_wxyz = pose_to_quat(T_book_init)
        book_xyzw = [book_q_wxyz[1], book_q_wxyz[2], book_q_wxyz[3], book_q_wxyz[0]]
        self._book_bid = sim.add_box(
            pos=book_pos,
            half_extents=self.book_size / 2,
            orn_xyzw=book_xyzw,
            rgba=(0.95, 0.75, 0.25, 1.0),
            mass=0.0,
        )

        # Obstacle (sphere, yellow mustard bottle)
        obs_pos = self._obstacle_pos
        cshape = pb.createCollisionShape(
            pb.GEOM_SPHERE, radius=self.obstacle_radius, physicsClientId=sim._pcid
        )
        vshape = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=self.obstacle_radius,
            rgbaColor=[0.95, 0.85, 0.10, 0.90], physicsClientId=sim._pcid,
        )
        self._obstacle_bid = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cshape,
            baseVisualShapeIndex=vshape,
            basePosition=obs_pos.tolist(),
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=sim._pcid,
        )
        sim._scene_bodies.append(self._obstacle_bid)

    def update_held_objects(self, sim: PandaSimEnv, T_ee: np.ndarray) -> None:
        if self._book_bid < 0:
            return
        T_book = T_ee @ self._T_ee_to_book
        pos, q_wxyz = pose_to_quat(T_book)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._book_bid, pos.tolist(), q_xyzw,
            physicsClientId=sim._pcid,
        )

    def sync_to_sim(self, sim: PandaSimEnv, reset_book: bool = False) -> None:
        if self._rack_bid < 0:
            self.spawn_in_sim(sim)
            return
        rack_pos, rack_q_wxyz = pose_to_quat(self.T_rack)
        rack_xyzw = [rack_q_wxyz[1], rack_q_wxyz[2], rack_q_wxyz[3], rack_q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._rack_bid, rack_pos.tolist(), rack_xyzw,
            physicsClientId=sim._pcid,
        )
        if self._obstacle_bid >= 0:
            pb.resetBasePositionAndOrientation(
                self._obstacle_bid, self._obstacle_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=sim._pcid,
            )
        if reset_book and self._book_bid >= 0:
            T_book_init = self.T_start @ self._T_ee_to_book
            book_pos, book_q_wxyz = pose_to_quat(T_book_init)
            book_xyzw = [book_q_wxyz[1], book_q_wxyz[2], book_q_wxyz[3], book_q_wxyz[0]]
            pb.resetBasePositionAndOrientation(
                self._book_bid, book_pos.tolist(), book_xyzw,
                physicsClientId=sim._pcid,
            )

    def set_variant(self, rack_pos=None, rack_euler=None, obstacle_pos=None, **_kwargs) -> None:
        if rack_pos is None:
            rack_pos = self.T_rack[:3, 3].copy()
        if rack_euler is None:
            from scipy.spatial.transform import Rotation as Rot
            rack_euler = Rot.from_matrix(self.T_rack[:3, :3]).as_euler('xyz')
        if obstacle_pos is None:
            obstacle_pos = self._obstacle_pos.copy()
        self._obstacle_pos = np.asarray(obstacle_pos, float)
        self._rebuild_poses(np.asarray(rack_pos, float), np.asarray(rack_euler, float))

    def get_start_q(self) -> np.ndarray:
        q, _ok, _err = ik(self.T_start, q0=Q_NEUTRAL)
        return q


def obstacle_scene_jitter(scene: ObstacleAvoidanceScene, demo_idx: int, rng: np.random.Generator) -> None:
    dp = rng.uniform(-0.01, 0.01, 3)
    dp[2] = 0.0
    new_rack = _DEFAULT_RACK_POS + dp
    # Also jitter obstacle slightly
    obs_dp = rng.uniform(-0.008, 0.008, 3)
    new_obs = _DEFAULT_OBSTACLE_POS + obs_dp
    scene.set_variant(rack_pos=new_rack, obstacle_pos=new_obs)


OBSTACLE_AVOIDANCE_TASK = TaskSpec(
    name="obstacle_avoidance",
    phases=[
        Phase(
            name="approach",
            goal_key="approach",
            n_steps=50,
            jitter_pos=0.02,
            jitter_z=0.015,
            arc_height=0.0,
        ),
        Phase(
            name="insert",
            goal_key="insert",
            n_steps=30,
            jitter_pos=0.008,
            arc_height=0.0,
        ),
    ],
    dt=0.01,
)


def make_ood_scenes() -> dict:
    return {
        "translate": ObstacleAvoidanceScene(
            rack_pos=np.array([0.55, 0.05, 0.15]),
            obstacle_pos=np.array([0.50, 0.00, 0.18]),
        ),
        "rotate": ObstacleAvoidanceScene(
            rack_pos=np.array([0.55, -0.10, 0.15]),
            rack_euler_xyz=np.array([0., 0., np.radians(20)]),
            obstacle_pos=np.array([0.50, -0.05, 0.18]),
        ),
        "close_obstacle": ObstacleAvoidanceScene(
            obstacle_pos=np.array([0.52, -0.08, 0.16]),
        ),
    }
