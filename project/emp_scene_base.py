"""
emp_scene_base.py — Abstract base class for EMP task scenes
============================================================
Subclass this for every task you want to run through the EMP pipeline.

A Scene owns:
  ① Geometry   — where objects live in the world
  ② Simulation — how to spawn rigid bodies in PyBullet
  ③ Kinematics — the EE starting pose + named phase-goal poses
  ④ Object tracking — teleporting held objects with the EE during demos
  ⑤ Variants   — reconfiguring geometry for OOD adaptation

The EMP pipeline (TaskSpec, emp_policy) knows nothing about scenes except
this interface.  Swapping a scene is the only change needed to switch tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class SceneBase(ABC):
    """
    Abstract task scene.

    Minimal contract for any scene that plugs into the EMP pipeline.

    Example
    -------
    class MyScene(SceneBase):
        def spawn_in_sim(self, sim): ...
        def get_start_T(self): return self.T_start
        def get_phase_goals(self): return {"goal_a": self.T_a, "goal_b": self.T_b}
        def set_variant(self, **kw): ...   # e.g. kw = {"target_pos": [...]}
    """

    # ── Must implement ────────────────────────────────────────────────────────

    @abstractmethod
    def spawn_in_sim(self, sim) -> None:
        """
        Create / refresh all rigid bodies in *sim* (a PandaSimEnv).

        Must be idempotent: calling twice removes old bodies first, then
        respawns them.  This allows the same scene object to be reused across
        multiple simulation episodes.
        """

    @abstractmethod
    def get_start_T(self) -> np.ndarray:
        """
        Return the 4×4 SE(3) EE pose at the START of a demo.

        This is the canonical starting transform.  TaskSpec applies per-demo
        jitter on top of it so each demonstration begins at a slightly
        different pose for training diversity.
        """

    @abstractmethod
    def get_phase_goals(self) -> Dict[str, np.ndarray]:
        """
        Return {phase_name: T_goal (4×4)} for every named phase.

        Keys must match the ``goal_key`` values declared in the TaskSpec's
        Phase list.  Example::

            return {
                "approach": self.T_approach,   # slot entrance
                "insert":   self.T_slot,        # fully inside
                "retract":  self.T_retract,     # pulled back
            }
        """

    @abstractmethod
    def set_variant(self, **kwargs) -> None:
        """
        Reconfigure scene geometry for cross-demo jitter or OOD adaptation.

        Scene-specific kwargs (e.g. rack_pos, rack_euler, target_pos).
        After calling this, get_start_T() and get_phase_goals() must reflect
        the new configuration.
        """

    # ── Override if scene has held / manipulated objects ──────────────────────

    def update_held_objects(self, sim, T_ee: np.ndarray) -> None:
        """
        Teleport any object the arm is holding so it tracks the EE.

        Called every demo step by TaskSpec.generate_demo().
        Default is a no-op; override when the robot is holding something.

        Parameters
        ----------
        sim  : PandaSimEnv
        T_ee : (4, 4) current EE world transform
        """

    # ── Convenience (may override) ────────────────────────────────────────────

    def get_goal_T(self) -> np.ndarray:
        """
        Return the canonical final goal pose (4×4) — the EMP attractor x*.

        Default: the last entry in get_phase_goals().
        Override this if your task's goal is not the final phase.
        """
        goals = self.get_phase_goals()
        if not goals:
            raise RuntimeError("get_phase_goals() returned an empty dict.")
        last_key = list(goals.keys())[-1]
        return goals[last_key].copy()

    def get_start_q(self) -> np.ndarray:
        """
        Return joint angles (7,) corresponding to get_start_T().

        Default: solves IK from Q_NEUTRAL.
        Override to cache a known-good solution.
        """
        from pb_stage1_env import ik, Q_NEUTRAL
        T_start = self.get_start_T()
        q, _ok, _err = ik(T_start, q0=Q_NEUTRAL)
        return q
