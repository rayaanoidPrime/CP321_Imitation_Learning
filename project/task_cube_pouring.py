"""
task_cube_pouring.py — Cube-pouring scene and task definition
==============================================================
Concrete SceneBase for the Franka Panda cube-pouring task.

Task description
----------------
The arm STARTS already holding a container filled with small cubes (e.g. a
saucepan or bowl).  It performs a 3-phase pouring motion:

    Phase 1 — "approach"  : Move from start pose above the saucepan rim
    Phase 2 — "pour"      : Tilt the container so cubes fall into the saucepan
    Phase 3 — "retract"   : Return the container to an upright, retracted pose

The container is a kinematic rigid body that is teleported to follow the EE
at every demo step, making it visible in the PyBullet viewer.

Usage
-----
    from task_cube_pouring import CubePourScene, CUBE_POUR_TASK

    scene = CubePourScene()
    scene.spawn_in_sim(sim)

    demos = generate_demonstrations(CUBE_POUR_TASK, scene, N_demos=7, sim=sim)

    # For adaptation, vary the saucepan position:
    scene.set_variant(pan_pos=np.array([0.55, 0.10, 0.04]))
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pybullet as pb

from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat, ik, Q_NEUTRAL
from emp_scene_base import SceneBase
from emp_task_spec import Phase, TaskSpec


# ══════════════════════════════════════════════════════════════════════════════
# CubePourScene
# ══════════════════════════════════════════════════════════════════════════════

#: Default saucepan centre position (world frame, table top at z=0)
_DEFAULT_PAN_POS = np.array([0.55, 0.10, 0.04])

#: Container (bowl) dimensions [diameter × diameter × height] in metres.
_CONTAINER_SIZE = np.array([0.10, 0.10, 0.06])

#: Offset from EE origin to container centre.
#: The container hangs below the EE, tilted slightly forward for pouring.
_T_EE_TO_CONTAINER_DEFAULT = make_pose(
    np.array([0.0, 0.0, -0.06]),
    euler_xyz=np.radians([0, 0, 0]),
)


class CubePourScene(SceneBase):
    """
    Cube-pouring scene.

    Geometry (world frame, table top at z = 0)
    -------------------------------------------
    Robot base   : [0, 0, 0]
    Saucepan     : pan_pos   (default [0.55, 0.10, 0.04])
    Pour goal    : centred above the saucepan, tilted ~60° forward
    Approach     : above the saucepan, upright
    Start pose   : higher and further back from the saucepan

    Parameters
    ----------
    pan_pos        : (3,) saucepan centre in world frame
    pan_euler_xyz  : (3,) saucepan rotation (Euler XYZ, radians)
    container_size : (3,) [diameter_x, diameter_y, height]  (metres)
    T_ee_to_container : (4,4) fixed transform from EE frame to container centre
    """

    def __init__(
            self,
            pan_pos:        np.ndarray = None,
            pan_euler_xyz:  np.ndarray = None,
            container_size: np.ndarray = None,
            T_ee_to_container: np.ndarray = None,
    ) -> None:
        self._pan_pos_default = (
            _DEFAULT_PAN_POS.copy() if pan_pos is None else np.asarray(pan_pos, float)
        )
        pan_euler = np.zeros(3) if pan_euler_xyz is None else np.asarray(pan_euler_xyz, float)

        self.container_size = (
            _CONTAINER_SIZE.copy() if container_size is None else np.asarray(container_size, float)
        )
        self._T_ee_to_container = (
            _T_EE_TO_CONTAINER_DEFAULT.copy()
            if T_ee_to_container is None
            else np.asarray(T_ee_to_container, float)
        )

        self._pan_bid: int = -1
        self._container_bid: int = -1

        self.T_pan: np.ndarray = None
        self.T_approach: np.ndarray = None
        self.T_pour: np.ndarray = None
        self.T_start: np.ndarray = None
        self.T_retract: np.ndarray = None

        self._rebuild_poses(self._pan_pos_default, pan_euler)

    def _rebuild_poses(self, pan_pos: np.ndarray, pan_euler: np.ndarray) -> None:
        """Recompute all derived SE(3) poses from saucepan position + orientation."""
        self.T_pan = make_pose(pan_pos, euler_xyz=pan_euler)

        # Approach: directly above the saucepan centre, upright orientation
        approach_offset = np.array([0.0, 0.0, 0.18])
        self.T_approach = self.T_pan @ make_pose(approach_offset)

        # Pour: lower and tilted forward (~60° pitch) to pour cubes into the pan
        pour_offset = np.array([0.0, 0.0, 0.10])
        self.T_pour = self.T_pan @ make_pose(pour_offset, euler_xyz=np.radians([-60, 0, 0]))

        # Start: higher and further back from the saucepan
        self.T_start = self.T_approach.copy()
        self.T_start[:3, 3] += np.array([-0.15, 0.0, 0.10])

        # Retract: lift the container back up and slightly back
        self.T_retract = self.T_approach.copy()
        self.T_retract[:3, 3] += np.array([-0.10, 0.0, 0.05])

    def get_start_T(self) -> np.ndarray:
        return self.T_start.copy()

    def get_phase_goals(self) -> Dict[str, np.ndarray]:
        return {
            "approach": self.T_approach.copy(),
            "pour":     self.T_pour.copy(),
            "retract":  self.T_retract.copy(),
        }

    def get_goal_T(self) -> np.ndarray:
        """The EMP attractor = the pour pose (the key motion endpoint)."""
        return self.T_pour.copy()

    def spawn_in_sim(self, sim: PandaSimEnv) -> None:
        # Remove stale bodies
        for bid in (self._pan_bid, self._container_bid):
            if bid >= 0:
                try:
                    pb.removeBody(bid, physicsClientId=sim._pcid)
                    if bid in sim._scene_bodies:
                        sim._scene_bodies.remove(bid)
                except Exception:
                    pass
        self._pan_bid = -1
        self._container_bid = -1

        # Saucepan — a shallow cylinder-like box (static)
        pan_pos, pan_q_wxyz = pose_to_quat(self.T_pan)
        pan_xyzw = [pan_q_wxyz[1], pan_q_wxyz[2], pan_q_wxyz[3], pan_q_wxyz[0]]
        self._pan_bid = sim.add_box(
            pos=pan_pos,
            half_extents=np.array([0.10, 0.10, 0.04]),
            orn_xyzw=pan_xyzw,
            rgba=(0.30, 0.30, 0.30, 0.90),
            mass=0.0,
        )

        # Container — starts at the EE's canonical start pose
        T_container_init = self.T_start @ self._T_ee_to_container
        cont_pos, cont_q_wxyz = pose_to_quat(T_container_init)
        cont_xyzw = [cont_q_wxyz[1], cont_q_wxyz[2], cont_q_wxyz[3], cont_q_wxyz[0]]
        self._container_bid = sim.add_box(
            pos=cont_pos,
            half_extents=self.container_size / 2,
            orn_xyzw=cont_xyzw,
            rgba=(0.85, 0.65, 0.25, 0.95),
            mass=0.0,
        )

    def update_held_objects(self, sim: PandaSimEnv, T_ee: np.ndarray) -> None:
        if self._container_bid < 0:
            return
        T_container = T_ee @ self._T_ee_to_container
        pos, q_wxyz = pose_to_quat(T_container)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._container_bid, pos.tolist(), q_xyzw,
            physicsClientId=sim._pcid,
        )

    def sync_to_sim(self, sim: PandaSimEnv, reset_container: bool = False) -> None:
        if self._pan_bid < 0:
            self.spawn_in_sim(sim)
            return
        pan_pos, pan_q_wxyz = pose_to_quat(self.T_pan)
        pan_xyzw = [pan_q_wxyz[1], pan_q_wxyz[2], pan_q_wxyz[3], pan_q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._pan_bid, pan_pos.tolist(), pan_xyzw,
            physicsClientId=sim._pcid,
        )
        if reset_container and self._container_bid >= 0:
            T_init = self.T_start @ self._T_ee_to_container
            cont_pos, cont_q_wxyz = pose_to_quat(T_init)
            cont_xyzw = [cont_q_wxyz[1], cont_q_wxyz[2], cont_q_wxyz[3], cont_q_wxyz[0]]
            pb.resetBasePositionAndOrientation(
                self._container_bid, cont_pos.tolist(), cont_xyzw,
                physicsClientId=sim._pcid,
            )

    def set_variant(self, pan_pos: np.ndarray = None, pan_euler: np.ndarray = None, **_kwargs) -> None:
        if pan_pos is None:
            pan_pos = self.T_pan[:3, 3].copy()
        if pan_euler is None:
            from scipy.spatial.transform import Rotation as Rot
            pan_euler = Rot.from_matrix(self.T_pan[:3, :3]).as_euler('xyz')
        self._rebuild_poses(np.asarray(pan_pos, float), np.asarray(pan_euler, float))

    def get_start_q(self) -> np.ndarray:
        q, _ok, _err = ik(self.T_start, q0=Q_NEUTRAL)
        return q


def cube_pan_jitter(scene: CubePourScene, demo_idx: int, rng: np.random.Generator) -> None:
    dp = rng.uniform(-0.012, 0.012, 3)
    dp[2] = 0.0
    new_pos = _DEFAULT_PAN_POS + dp
    scene.set_variant(pan_pos=new_pos)


CUBE_POUR_TASK = TaskSpec(
    name="cube_pour",
    phases=[
        Phase(
            name="approach",
            goal_key="approach",
            n_steps=45,
            jitter_pos=0.02,
            jitter_z=0.015,
            arc_height=0.0,
        ),
        Phase(
            name="pour",
            goal_key="pour",
            n_steps=35,
            jitter_pos=0.01,
            arc_height=0.0,
        ),
        Phase(
            name="retract",
            goal_key="retract",
            n_steps=20,
            jitter_pos=0.015,
            arc_height=0.05,
        ),
    ],
    dt=0.01,
)


def make_ood_scenes() -> dict:
    return {
        "translate": CubePourScene(
            pan_pos=np.array([0.55, 0.20, 0.04]),
        ),
        "rotate": CubePourScene(
            pan_pos=np.array([0.55, 0.10, 0.04]),
            pan_euler_xyz=np.array([0., 0., np.radians(20)]),
        ),
        "full_ood": CubePourScene(
            pan_pos=np.array([0.60, 0.15, 0.06]),
            pan_euler_xyz=np.array([0., 0., np.radians(-15)]),
        ),
    }
