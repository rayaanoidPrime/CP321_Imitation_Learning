"""
task_multi_pick_place.py — Multi-step pick-and-place scene and task definition
===============================================================================
Concrete SceneBase for the Franka Panda multi-step pick-and-place task.

Task description
----------------
The arm performs multiple pick-and-place motions in sequence WITHOUT releasing
the object (the object is kinematically attached to the EE throughout).

    Phase 1 — "pick_1"     : Move to first object location
    Phase 2 — "place_1"    : Move to first target location
    Phase 3 — "pick_2"     : Move to second object location  (arm still holding)
    Phase 4 — "place_2"    : Move to second target location
    Phase 5 — "retract"    : Pull back to home position

The object is a kinematic rigid body that is teleported to follow the EE
at every demo step, making it visible in the PyBullet viewer.

Usage
-----
    from task_multi_pick_place import MultiPickPlaceScene, MULTI_PICK_PLACE_TASK

    scene = MultiPickPlaceScene()
    scene.spawn_in_sim(sim)

    demos = generate_demonstrations(MULTI_PICK_PLACE_TASK, scene, N_demos=7, sim=sim)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pybullet as pb

from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat, ik, Q_NEUTRAL
from emp_scene_base import SceneBase
from emp_task_spec import Phase, TaskSpec


# ══════════════════════════════════════════════════════════════════════════════
# MultiPickPlaceScene
# ══════════════════════════════════════════════════════════════════════════════

#: Default positions (world frame, table top at z=0)
_DEFAULT_PICK_1_POS = np.array([0.45, -0.15, 0.02])
_DEFAULT_PLACE_1_POS = np.array([0.55, 0.15, 0.02])
_DEFAULT_PICK_2_POS = np.array([0.40, 0.10, 0.02])
_DEFAULT_PLACE_2_POS = np.array([0.60, -0.10, 0.02])

#: Object (small box) dimensions in metres
_OBJECT_SIZE = np.array([0.04, 0.04, 0.04])

#: Offset from EE origin to object centre when held
_T_EE_TO_OBJECT_DEFAULT = make_pose(np.array([0.0, 0.0, -0.04]))


class MultiPickPlaceScene(SceneBase):
    """
    Multi-step pick-and-place scene.

    Geometry (world frame, table top at z = 0)
    -------------------------------------------
    Robot base   : [0, 0, 0]
    Pick 1       : pick_1_pos  (first object location)
    Place 1      : place_1_pos (first target)
    Pick 2       : pick_2_pos  (second object location)
    Place 2      : place_2_pos (second target)
    Start pose   : above pick_1, ready to descend

    Parameters
    ----------
    pick_1_pos   : (3,) first pick location
    place_1_pos  : (3,) first place location
    pick_2_pos   : (3,) second pick location
    place_2_pos  : (3,) second place location
    euler_z      : float rotation around Z for all locations
    object_size  : (3,) object dimensions
    T_ee_to_object : (4,4) fixed transform from EE to held object centre
    """

    def __init__(
            self,
            pick_1_pos:    np.ndarray = None,
            place_1_pos:   np.ndarray = None,
            pick_2_pos:    np.ndarray = None,
            place_2_pos:   np.ndarray = None,
            euler_z:       float = 0.0,
            object_size:   np.ndarray = None,
            T_ee_to_object: np.ndarray = None,
    ) -> None:
        self._pick_1_default = _DEFAULT_PICK_1_POS.copy() if pick_1_pos is None else np.asarray(pick_1_pos, float)
        self._place_1_default = _DEFAULT_PLACE_1_POS.copy() if place_1_pos is None else np.asarray(place_1_pos, float)
        self._pick_2_default = _DEFAULT_PICK_2_POS.copy() if pick_2_pos is None else np.asarray(pick_2_pos, float)
        self._place_2_default = _DEFAULT_PLACE_2_POS.copy() if place_2_pos is None else np.asarray(place_2_pos, float)

        self._euler_z = euler_z

        self.object_size = _OBJECT_SIZE.copy() if object_size is None else np.asarray(object_size, float)
        self._T_ee_to_object = (
            _T_EE_TO_OBJECT_DEFAULT.copy()
            if T_ee_to_object is None
            else np.asarray(T_ee_to_object, float)
        )

        self._object_bid: int = -1
        self._marker_bids: list = []

        self.T_pick_1: np.ndarray = None
        self.T_place_1: np.ndarray = None
        self.T_pick_2: np.ndarray = None
        self.T_place_2: np.ndarray = None
        self.T_start: np.ndarray = None
        self.T_retract: np.ndarray = None

        self._rebuild_poses(
            self._pick_1_default,
            self._place_1_default,
            self._pick_2_default,
            self._place_2_default,
        )

    def _rebuild_poses(self, pick_1, place_1, pick_2, place_2):
        """Recompute all derived SE(3) poses from pick/place positions."""
        Rz = make_pose(np.zeros(3), euler_xyz=np.array([0, 0, self._euler_z]))

        def _pose_at(p):
            """Create an EE pose at position p, pointing downward (z-axis down)."""
            # EE points down so the held object is below the gripper
            T = make_pose(p, euler_xyz=np.radians([180, 0, 0]))
            T[:3, :3] = T[:3, :3] @ Rz[:3, :3]
            return T

        # Pick poses: EE descends to object location, pointing down
        self.T_pick_1 = _pose_at(pick_1)
        self.T_pick_2 = _pose_at(pick_2)

        # Place poses: EE at target location, pointing down
        self.T_place_1 = _pose_at(place_1)
        self.T_place_2 = _pose_at(place_2)

        # Start: above pick_1, upright orientation (ready to descend)
        self.T_start = _pose_at(pick_1)
        self.T_start[:3, 3] += np.array([0.0, 0.0, 0.15])
        # Upright orientation at start
        self.T_start[:3, :3] = make_pose(np.zeros(3), euler_xyz=np.array([0, 0, self._euler_z]))[:3, :3]

        # Retract: above and behind, upright
        self.T_retract = self.T_start.copy()
        self.T_retract[:3, 3] += np.array([-0.10, 0.0, 0.05])

    def get_start_T(self) -> np.ndarray:
        return self.T_start.copy()

    def get_phase_goals(self) -> Dict[str, np.ndarray]:
        return {
            "pick_1":   self.T_pick_1.copy(),
            "place_1":  self.T_place_1.copy(),
            "pick_2":   self.T_pick_2.copy(),
            "place_2":  self.T_place_2.copy(),
            "retract":  self.T_retract.copy(),
        }

    def get_goal_T(self) -> np.ndarray:
        """The EMP attractor = the final place pose."""
        return self.T_place_2.copy()

    def spawn_in_sim(self, sim: PandaSimEnv) -> None:
        # Remove stale bodies
        if self._object_bid >= 0:
            try:
                pb.removeBody(self._object_bid, physicsClientId=sim._pcid)
                if self._object_bid in sim._scene_bodies:
                    sim._scene_bodies.remove(self._object_bid)
            except Exception:
                pass
        for bid in self._marker_bids:
            try:
                pb.removeBody(bid, physicsClientId=sim._pcid)
                if bid in sim._scene_bodies:
                    sim._scene_bodies.remove(bid)
            except Exception:
                pass
        self._object_bid = -1
        self._marker_bids = []

        # Object — starts at the EE's canonical start pose
        T_obj_init = self.T_start @ self._T_ee_to_object
        obj_pos, obj_q_wxyz = pose_to_quat(T_obj_init)
        obj_xyzw = [obj_q_wxyz[1], obj_q_wxyz[2], obj_q_wxyz[3], obj_q_wxyz[0]]
        self._object_bid = sim.add_box(
            pos=obj_pos,
            half_extents=self.object_size / 2,
            orn_xyzw=obj_xyzw,
            rgba=(0.20, 0.70, 0.20, 0.95),
            mass=0.0,
        )

        # Location markers (small spheres)
        marker_positions = [
            (self.T_pick_1[:3, 3],   (0.9, 0.3, 0.1, 0.7)),
            (self.T_place_1[:3, 3],  (0.1, 0.3, 0.9, 0.7)),
            (self.T_pick_2[:3, 3],   (0.9, 0.6, 0.1, 0.7)),
            (self.T_place_2[:3, 3],  (0.1, 0.6, 0.9, 0.7)),
        ]
        for pos, rgba in marker_positions:
            cshape = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=0.008, physicsClientId=sim._pcid
            )
            vshape = pb.createVisualShape(
                pb.GEOM_SPHERE, radius=0.008,
                rgbaColor=list(rgba), physicsClientId=sim._pcid,
            )
            bid = pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=cshape,
                baseVisualShapeIndex=vshape,
                basePosition=pos.tolist(),
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=sim._pcid,
            )
            self._marker_bids.append(bid)

    def update_held_objects(self, sim: PandaSimEnv, T_ee: np.ndarray) -> None:
        if self._object_bid < 0:
            return
        T_obj = T_ee @ self._T_ee_to_object
        pos, q_wxyz = pose_to_quat(T_obj)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        pb.resetBasePositionAndOrientation(
            self._object_bid, pos.tolist(), q_xyzw,
            physicsClientId=sim._pcid,
        )

    def set_variant(self, pick_1_pos=None, place_1_pos=None,
                    pick_2_pos=None, place_2_pos=None, euler_z=None, **_kwargs) -> None:
        if pick_1_pos is None:
            pick_1_pos = self.T_pick_1[:3, 3].copy()
        if place_1_pos is None:
            place_1_pos = self.T_place_1[:3, 3].copy()
        if pick_2_pos is None:
            pick_2_pos = self.T_pick_2[:3, 3].copy()
        if place_2_pos is None:
            place_2_pos = self.T_place_2[:3, 3].copy()
        if euler_z is None:
            euler_z = self._euler_z
        self._euler_z = euler_z
        self._rebuild_poses(
            np.asarray(pick_1_pos, float),
            np.asarray(place_1_pos, float),
            np.asarray(pick_2_pos, float),
            np.asarray(place_2_pos, float),
        )

    def get_start_q(self) -> np.ndarray:
        q, _ok, _err = ik(self.T_start, q0=Q_NEUTRAL)
        return q


def multi_pick_place_jitter(scene: MultiPickPlaceScene, demo_idx: int, rng: np.random.Generator) -> None:
    dp1 = rng.uniform(-0.01, 0.01, 3); dp1[2] = 0.0
    dp2 = rng.uniform(-0.01, 0.01, 3); dp2[2] = 0.0
    dp3 = rng.uniform(-0.01, 0.01, 3); dp3[2] = 0.0
    dp4 = rng.uniform(-0.01, 0.01, 3); dp4[2] = 0.0
    scene.set_variant(
        pick_1_pos=_DEFAULT_PICK_1_POS + dp1,
        place_1_pos=_DEFAULT_PLACE_1_POS + dp2,
        pick_2_pos=_DEFAULT_PICK_2_POS + dp3,
        place_2_pos=_DEFAULT_PLACE_2_POS + dp4,
    )


MULTI_PICK_PLACE_TASK = TaskSpec(
    name="multi_pick_place",
    phases=[
        Phase(
            name="pick_1",
            goal_key="pick_1",
            n_steps=35,
            jitter_pos=0.015,
            jitter_z=0.01,
            arc_height=0.06,
        ),
        Phase(
            name="place_1",
            goal_key="place_1",
            n_steps=35,
            jitter_pos=0.015,
            jitter_z=0.01,
            arc_height=0.06,
        ),
        Phase(
            name="pick_2",
            goal_key="pick_2",
            n_steps=35,
            jitter_pos=0.015,
            jitter_z=0.01,
            arc_height=0.06,
        ),
        Phase(
            name="place_2",
            goal_key="place_2",
            n_steps=35,
            jitter_pos=0.015,
            jitter_z=0.01,
            arc_height=0.06,
        ),
        Phase(
            name="retract",
            goal_key="retract",
            n_steps=20,
            jitter_pos=0.01,
            arc_height=0.0,
        ),
    ],
    dt=0.01,
)


def make_ood_scenes() -> dict:
    return {
        "translate": MultiPickPlaceScene(
            pick_1_pos=np.array([0.45, -0.10, 0.02]),
            place_1_pos=np.array([0.55, 0.20, 0.02]),
            pick_2_pos=np.array([0.40, 0.15, 0.02]),
            place_2_pos=np.array([0.60, -0.05, 0.02]),
        ),
        "rotate": MultiPickPlaceScene(
            euler_z=np.radians(25),
        ),
        "full_ood": MultiPickPlaceScene(
            pick_1_pos=np.array([0.50, -0.05, 0.02]),
            place_1_pos=np.array([0.60, 0.20, 0.02]),
            pick_2_pos=np.array([0.45, 0.15, 0.02]),
            place_2_pos=np.array([0.65, -0.15, 0.02]),
            euler_z=np.radians(-15),
        ),
    }
