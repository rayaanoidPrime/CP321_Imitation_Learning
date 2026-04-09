"""
emp_task_spec.py — Task specification, demo generation, save / load
====================================================================
A TaskSpec is a named, ordered list of motion Phases.
Each Phase maps to one named goal pose in the scene and controls how the
arm interpolates to that pose.

Multi-step tasks are defined by simply adding more phases::

    PLACE_THEN_SLIDE = TaskSpec("place_slide", [
        Phase("to_slot_a", "slot_a", n_steps=40),
        Phase("to_slot_b", "slot_b", n_steps=35),   # arm moves on without releasing
        Phase("retract",   "retract", n_steps=15),
    ])

Key entry points
----------------
TaskSpec.generate_waypoints(scene, rng)    → list of (T, n_steps)
TaskSpec.generate_demo(scene, sim, rng)    → demo dict
generate_demonstrations(task, scene, ...)  → list of demo dicts
save_demos / load_demos
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from pb_stage1_env import (
    fk, ik, Q_NEUTRAL, se3_interpolate, pose_to_quat,
    quat_mul, quat_conjugate, quat_log,
)
from emp_scene_base import SceneBase


# ══════════════════════════════════════════════════════════════════════════════
# Phase — one motion segment
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phase:
    """
    One motion segment in a TaskSpec.

    Parameters
    ----------
    name       : Human-readable identifier ("approach", "insert", …).
    goal_key   : Key into SceneBase.get_phase_goals() that gives T_goal.
    n_steps    : Number of SE(3) interpolation steps for this segment.
    jitter_pos : Uniform ±noise (m) applied to goal XYZ each demo.
    jitter_z   : If > 0, replaces the Z component of jitter_pos noise.
    arc_height : If > 0, a via-point is inserted above the goal before
                 descending to it.  Useful for "lift-and-swing" motions.
                 For straight-line insertions set to 0.
    """
    name:       str
    goal_key:   str
    n_steps:    int
    jitter_pos: float = 0.0
    jitter_z:   float = 0.0
    arc_height: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TaskSpec — ordered sequence of phases
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSpec:
    """
    A complete task description: name + ordered list of Phases + timing.

    Typical usage
    -------------
    task = TaskSpec("book_insert", [
        Phase("approach", "approach", n_steps=40, jitter_pos=0.025),
        Phase("insert",   "insert",   n_steps=30),
        Phase("retract",  "retract",  n_steps=15),
    ])

    # Generate waypoints for a scene (pure SE(3), no IK):
    wpts = task.generate_waypoints(scene, rng)

    # Generate one full demo (runs IK at every step):
    demo = task.generate_demo(scene, sim=sim, rng=rng)

    # Generate many demos at once:
    demos = generate_demonstrations(task, scene, N_demos=7)
    """
    name:   str
    phases: List[Phase]
    dt:     float = 0.01   # seconds per interpolation step

    @staticmethod
    def _normalise_quat_sequence(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=float).copy()
        if len(quat) == 0:
            return quat
        quat[0] /= np.linalg.norm(quat[0])
        for i in range(1, len(quat)):
            if np.dot(quat[i - 1], quat[i]) < 0.0:
                quat[i] *= -1.0
            quat[i] /= np.linalg.norm(quat[i])
        return quat

    # ── Waypoint list ─────────────────────────────────────────────────────────

    def generate_waypoints(
            self,
            scene: SceneBase,
            rng:   np.random.Generator = None,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Build [(T_waypoint (4×4), n_steps), …] for all phases.

        The list always starts with (T_start, 1).  Subsequent entries are the
        jittered goal poses for each phase.  Phases with arc_height > 0 expand
        into two entries (via-point, then goal).

        Parameters
        ----------
        scene : SceneBase (must be configured before calling)
        rng   : random generator (created from default seed if None)

        Returns
        -------
        List of (T (4×4), n_steps) tuples.
        """
        if rng is None:
            rng = np.random.default_rng()

        goals = scene.get_phase_goals()
        for phase in self.phases:
            if phase.goal_key not in goals:
                raise KeyError(
                    f"Phase '{phase.name}' uses goal_key='{phase.goal_key}' "
                    f"but scene only has: {list(goals.keys())}"
                )

        # ── Jittered start pose ────────────────────────────────────────────
        T_start = scene.get_start_T()
        T_s = T_start.copy()
        dp = rng.uniform(-0.025, 0.025, 3)
        dp[2] = rng.uniform(-0.015, 0.015)
        T_s[:3, 3] += dp

        waypoints: List[Tuple[np.ndarray, int]] = [(T_s, 1)]

        # ── Per-phase waypoints ────────────────────────────────────────────
        for phase in self.phases:
            T_goal = goals[phase.goal_key].copy()

            # Positional jitter on this phase's goal
            if phase.jitter_pos > 0:
                dp = rng.uniform(-phase.jitter_pos, phase.jitter_pos, 3)
                if phase.jitter_z > 0:
                    dp[2] = rng.uniform(-phase.jitter_z, phase.jitter_z)
                T_goal[:3, 3] += dp

            # Optional arc via-point (good for "lift over obstacle" motions)
            if phase.arc_height > 0:
                T_via = T_goal.copy()
                T_via[:3, 3] += np.array(
                    [0.0, 0.0, phase.arc_height + rng.uniform(-0.01, 0.02)]
                )
                n_up   = phase.n_steps // 2
                n_down = phase.n_steps - n_up
                waypoints.append((T_via,   n_up))
                waypoints.append((T_goal,  n_down))
            else:
                waypoints.append((T_goal, phase.n_steps))

        return waypoints

    # ── Single demo ───────────────────────────────────────────────────────────

    def generate_demo(
            self,
            scene: SceneBase,
            sim          = None,
            rng:           np.random.Generator = None,
    ) -> dict:
        """
        Run one complete demo: SE(3) interpolation → IK → record.

        If *sim* is provided, `scene.update_held_objects(sim, T_ee)` is called
        at every step, so any object the arm holds tracks the EE in the physics
        world.

        Parameters
        ----------
        scene : SceneBase
        sim   : PandaSimEnv (optional, only needed for live visual)
        rng   : random generator

        Returns
        -------
        dict with keys:
            pos          : (T, 3)  EE positions
            quat         : (T, 4)  EE quaternions (wxyz)
            vel          : (T, 3)  EE linear velocity
            ang_vel      : (T, 3)  EE angular velocity
            q_joints     : (T, 7)  joint angles
            times        : (T,)    timestamps (s)
            T_goal       : (4, 4)  canonical final goal (= EMP attractor)
            phase_names  : list[str] — names of phases in order
            phase_starts : list[int] — trajectory index where each phase begins
        """
        if rng is None:
            rng = np.random.default_rng()

        waypoints = self.generate_waypoints(scene, rng)

        poses:   List[np.ndarray] = []
        q_traj:  List[np.ndarray] = []
        times:   List[float]      = []
        t = 0.0
        q_cur = scene.get_start_q().copy()
        prev_target_pos: Optional[np.ndarray] = None

        # Track phase start indices in the trajectory
        # waypoints[0] = start (1 step), waypoints[1..] = phase goals
        phase_names:  List[str] = []
        phase_starts: List[int] = []

        # Map waypoint segment index → phase index (accounting for arc splits)
        seg_to_phase: Dict[int, int] = {}
        wpt_idx = 1   # skip the start entry
        for p_idx, phase in enumerate(self.phases):
            if phase.arc_height > 0:
                seg_to_phase[wpt_idx]     = p_idx   # arc via-point segment
                seg_to_phase[wpt_idx + 1] = p_idx   # descent segment
                wpt_idx += 2
            else:
                seg_to_phase[wpt_idx] = p_idx
                wpt_idx += 1

        # ── Interpolate ───────────────────────────────────────────────────
        for seg_idx in range(len(waypoints) - 1):
            T_a, _       = waypoints[seg_idx]
            T_b, n_steps = waypoints[seg_idx + 1]

            # Record phase boundary (first segment of each phase, not arc mid-points)
            p_idx = seg_to_phase.get(seg_idx + 1)
            if p_idx is not None:
                pname = self.phases[p_idx].name
                is_first_seg = (
                    (self.phases[p_idx].arc_height > 0 and seg_idx + 1 == list(seg_to_phase.keys())[0])
                    or (self.phases[p_idx].arc_height == 0)
                    or pname not in phase_names
                )
                if pname not in phase_names:
                    phase_names.append(pname)
                    phase_starts.append(len(poses))

            for k in range(n_steps):
                s    = k / max(n_steps - 1, 1)
                s_mj = s**3 * (10 - 15*s + 6*s**2)   # minimum-jerk profile
                T_k  = se3_interpolate(T_a, T_b, s_mj)

                q_k, ok, err = ik(T_k, q0=q_cur, tol=5e-4)
                if ok:
                    T_fk_cand, _ = fk(q_k)
                    prev_pos = poses[-1][:3, 3] if poses else None
                    desired_step = (
                        np.linalg.norm(T_k[:3, 3] - prev_target_pos)
                        if prev_target_pos is not None else 0.0
                    )
                    actual_step = (np.linalg.norm(T_fk_cand[:3, 3] - prev_pos)
                                   if prev_pos is not None else 0.0)
                    jump_limit = max(0.025, 3.0 * desired_step)
                    if prev_pos is None or actual_step <= jump_limit:
                        q_cur = q_k
                        T_fk = T_fk_cand
                    else:
                        alpha = max(0.1, min(1.0, jump_limit / actual_step))
                        q_cur = q_cur + alpha * (q_k - q_cur)
                        T_fk, _ = fk(q_cur)
                else:
                    T_fk, _ = fk(q_cur)

                poses.append(T_fk)
                q_traj.append(q_cur.copy())
                times.append(t)
                t += self.dt
                prev_target_pos = T_k[:3, 3].copy()

                # Teleport held objects in the physics world
                if sim is not None:
                    scene.update_held_objects(sim, T_fk)

        # ── Convert to arrays ──────────────────────────────────────────────
        pos      = np.array([T[:3, 3]          for T in poses])
        quat     = np.array([pose_to_quat(T)[1] for T in poses])   # wxyz
        quat     = self._normalise_quat_sequence(quat)
        times_a  = np.array(times)

        vel      = np.gradient(pos, axis=0) / self.dt

        ang_vel  = np.zeros((len(poses), 3))
        for i in range(1, len(poses)):
            dq = quat_mul(quat[i], quat_conjugate(quat[i - 1]))
            if dq[0] < 0.0:
                dq *= -1.0
            ang_vel[i] = quat_log(dq) / self.dt

        return dict(
            pos          = pos,
            quat         = quat,
            vel          = vel,
            ang_vel      = ang_vel,
            q_joints     = np.array(q_traj),
            times        = times_a,
            T_goal       = scene.get_goal_T(),          # 4×4 EMP attractor
            phase_names  = phase_names,
            phase_starts = phase_starts,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Batch demo generation
# ══════════════════════════════════════════════════════════════════════════════

def check_collision(sim):
    import pybullet as pb
    pts = pb.getContactPoints(physicsClientId=sim._pcid)
    return len(pts) > 0

def generate_demonstrations(
        task_spec:        TaskSpec,
        scene:            SceneBase,
        N_demos:          int   = 7,
        seed:             int   = 42,
        sim                     = None,
        noise_pos:        float = 0.003,
        noise_quat:       float = 0.005,
        scene_variant_fn: Optional[Callable] = None,
) -> List[dict]:
    """
    Generate N_demos demonstrations for any TaskSpec + SceneBase pair.

    Parameters
    ----------
    task_spec        : TaskSpec instance
    scene            : SceneBase instance (will be mutated if scene_variant_fn set)
    N_demos          : number of demonstrations
    seed             : master random seed (reproducible)
    sim              : PandaSimEnv — enables live held-object tracking
    noise_pos        : std of Gaussian position noise added post-hoc
    noise_quat       : std of Gaussian quaternion noise added post-hoc
    scene_variant_fn : callable(scene, demo_index, rng) that reconfigures the
                       scene for each demo.  Use this for rack-jitter, new
                       target positions, etc.

                       Example::

                           def book_jitter(scene, i, rng):
                               dp = rng.uniform(-0.01, 0.01, 3); dp[2]=0
                               scene.set_variant(rack_pos=BASE_POS + dp)

    Returns
    -------
    List of demo dicts, one per demonstration.
    """
    rng = np.random.default_rng(seed)
    demos = []

    while len(demos) < N_demos:
        if scene_variant_fn is not None:
            scene_variant_fn(scene, len(demos), rng) # keep diversity
        if sim is not None:
            scene.spawn_in_sim(sim)

        demo = task_spec.generate_demo(scene, sim=sim, rng=rng)

        collision = False
        if sim is not None:
            for q in demo['q_joints']:
                sim.set_joint_angles(q)
                if check_collision(sim):
                    collision = True
                    break

        if not collision:
            demos.append(demo)
            print(f"[OK] Accepted demo {len(demos)}")
        else:
            print("[WARN] Rejected demo (collision)")

    return demos


# ══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ══════════════════════════════════════════════════════════════════════════════

def save_demos(demos: List[dict], path: str = 'demos.npz') -> None:
    """
    Save demos list to a compressed numpy archive.

    Only numpy-array fields are serialised.  phase_names / phase_starts are
    saved as object arrays so they survive the round-trip.
    """
    arrays: Dict[str, np.ndarray] = {}
    for i, d in enumerate(demos):
        for k, v in d.items():
            key = f'demo{i}_{k}'
            if isinstance(v, np.ndarray):
                arrays[key] = v
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                try:
                    arrays[key] = np.array(v)
                except Exception:
                    pass   # skip non-serialisable fields
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(demos)} demos to {path}")


def load_demos(path: str) -> List[dict]:
    """
    Load demos from a .npz archive created by save_demos.

    Returns a list of dicts with the same schema as generate_demo output
    (minus non-array fields).
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # Discover demo indices
    indices = set()
    for k in keys:
        if k.startswith('demo'):
            try:
                idx = int(k.split('_')[0].replace('demo', ''))
                indices.add(idx)
            except ValueError:
                pass

    demos = []
    for i in sorted(indices):
        prefix = f'demo{i}_'
        d = {k[len(prefix):]: data[k] for k in keys if k.startswith(prefix)}
        demos.append(d)

    return demos
