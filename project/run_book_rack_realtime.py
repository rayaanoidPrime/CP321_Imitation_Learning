"""
run_book_rack_realtime.py - Realtime EMP adaptation with an interactive goal.

This runner keeps adapting a trained EMP policy while the rack goal moves.
You can move the goal with either:
1. Mouse: PyBullet debug sliders (goal x / y / z / yaw)
2. Keyboard: arrow keys + U/J + Q/E

Controls
--------
Arrow Up/Down   : goal x +/-
Arrow Left/Right: goal y +/-
U / J           : goal z +/-
Q / E           : goal yaw +/-
R               : reset rack pose and robot start
X               : exit loop
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pybullet as pb

from emp_policy import adapt_emp, rollout_se3
from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat
from run_book_rack import step1_demos, step2_train
from task_book_rack import BookInsertScene


@dataclass
class GoalState:
    pos: np.ndarray
    yaw: float


def _is_down(keys: dict, key: int) -> bool:
    return key in keys and (keys[key] & pb.KEY_IS_DOWN) != 0


def _was_triggered(keys: dict, key: int) -> bool:
    return key in keys and (keys[key] & pb.KEY_WAS_TRIGGERED) != 0


def _wrap_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _spawn_goal_marker(sim: PandaSimEnv, pos: np.ndarray) -> int:
    cshape = pb.createCollisionShape(
        pb.GEOM_SPHERE, radius=0.015, physicsClientId=sim._pcid
    )
    vshape = pb.createVisualShape(
        pb.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[0.90, 0.20, 0.20, 0.85],
        physicsClientId=sim._pcid,
    )
    return pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cshape,
        baseVisualShapeIndex=vshape,
        basePosition=pos.tolist(),
        baseOrientation=[0, 0, 0, 1],
        physicsClientId=sim._pcid,
    )


def _update_goal_marker(sim: PandaSimEnv, marker_bid: int, pos: np.ndarray) -> None:
    pb.resetBasePositionAndOrientation(
        marker_bid, pos.tolist(), [0, 0, 0, 1], physicsClientId=sim._pcid
    )


def _make_goal_sliders(sim: PandaSimEnv, state: GoalState) -> dict:
    return {
        "x": pb.addUserDebugParameter(
            "goal_x", 0.40, 0.80, float(state.pos[0]), physicsClientId=sim._pcid
        ),
        "y": pb.addUserDebugParameter(
            "goal_y", -0.30, 0.30, float(state.pos[1]), physicsClientId=sim._pcid
        ),
        "z": pb.addUserDebugParameter(
            "goal_z", 0.08, 0.30, float(state.pos[2]), physicsClientId=sim._pcid
        ),
        "yaw": pb.addUserDebugParameter(
            "goal_yaw_deg",
            -75.0,
            75.0,
            float(np.degrees(state.yaw)),
            physicsClientId=sim._pcid,
        ),
    }


def _read_goal_sliders(sim: PandaSimEnv, sliders: dict) -> GoalState:
    x = float(pb.readUserDebugParameter(sliders["x"], physicsClientId=sim._pcid))
    y = float(pb.readUserDebugParameter(sliders["y"], physicsClientId=sim._pcid))
    z = float(pb.readUserDebugParameter(sliders["z"], physicsClientId=sim._pcid))
    yaw_deg = float(pb.readUserDebugParameter(sliders["yaw"], physicsClientId=sim._pcid))
    return GoalState(pos=np.array([x, y, z], dtype=float), yaw=np.radians(yaw_deg))


def _apply_keyboard_goal_delta(keys: dict, state: GoalState, step_xyz: float, step_yaw: float) -> None:
    if _is_down(keys, pb.B3G_UP_ARROW):
        state.pos[0] += step_xyz
    if _is_down(keys, pb.B3G_DOWN_ARROW):
        state.pos[0] -= step_xyz
    if _is_down(keys, pb.B3G_LEFT_ARROW):
        state.pos[1] += step_xyz
    if _is_down(keys, pb.B3G_RIGHT_ARROW):
        state.pos[1] -= step_xyz
    if _is_down(keys, ord("u")) or _is_down(keys, ord("U")):
        state.pos[2] += step_xyz
    if _is_down(keys, ord("j")) or _is_down(keys, ord("J")):
        state.pos[2] -= step_xyz
    if _is_down(keys, ord("q")) or _is_down(keys, ord("Q")):
        state.yaw += step_yaw
    if _is_down(keys, ord("e")) or _is_down(keys, ord("E")):
        state.yaw -= step_yaw

    state.pos[0] = np.clip(state.pos[0], 0.40, 0.80)
    state.pos[1] = np.clip(state.pos[1], -0.30, 0.30)
    state.pos[2] = np.clip(state.pos[2], 0.08, 0.30)
    state.yaw = _wrap_pi(state.yaw)


def _goal_to_scene(scene: BookInsertScene, state: GoalState) -> None:
    scene.set_variant(
        rack_pos=state.pos,
        rack_euler=np.array([0.0, 0.0, state.yaw], dtype=float),
    )


def run_realtime(args: argparse.Namespace) -> None:
    with PandaSimEnv(gui=True) as sim:
        demos, scene = step1_demos(sim=sim)
        model = step2_train(demos)

        scene.spawn_in_sim(sim)
        q_start = scene.get_start_q().copy()
        sim.set_joint_angles(q_start)
        sim.step(3)

        rack_pos0 = scene.T_rack[:3, 3].copy()
        yaw0 = float(np.arctan2(scene.T_rack[1, 0], scene.T_rack[0, 0]))
        goal_state = GoalState(pos=rack_pos0.copy(), yaw=yaw0)

        sliders = _make_goal_sliders(sim, goal_state)
        slider_goal_prev = _read_goal_sliders(sim, sliders)
        marker_bid = _spawn_goal_marker(sim, scene.get_goal_T()[:3, 3])

        print("\nRealtime EMP loop started.")
        print("Use sliders (mouse) or keys: arrows, U/J, Q/E. R reset, X exit.")

        cached_plan_pos = np.zeros((0, 3), dtype=float)
        cached_plan_quat = np.zeros((0, 4), dtype=float)
        plan_idx = 0

        last_goal_pos = scene.get_goal_T()[:3, 3].copy()
        last_goal_quat = pose_to_quat(scene.get_goal_T())[1]
        last_applied_goal = GoalState(pos=goal_state.pos.copy(), yaw=goal_state.yaw)
        last_adapt_time = 0.0
        last_report_time = 0.0
        last_adapt_ms = 0.0
        loop_dt = 1.0 / max(args.loop_hz, 1.0)

        while True:
            t_loop = time.perf_counter()
            keys = pb.getKeyboardEvents(physicsClientId=sim._pcid)

            if _was_triggered(keys, ord("x")) or _was_triggered(keys, ord("X")):
                break

            if _was_triggered(keys, ord("r")) or _was_triggered(keys, ord("R")):
                goal_state = GoalState(pos=rack_pos0.copy(), yaw=yaw0)
                _goal_to_scene(scene, goal_state)
                scene.sync_to_sim(sim, reset_book=True)
                q_start = scene.get_start_q().copy()
                sim.set_joint_angles(q_start)
                cached_plan_pos = np.zeros((0, 3), dtype=float)
                cached_plan_quat = np.zeros((0, 4), dtype=float)
                plan_idx = 0
                print("Reset scene and robot.")

            slider_goal = _read_goal_sliders(sim, sliders)
            slider_delta = np.linalg.norm(slider_goal.pos - slider_goal_prev.pos) + abs(
                slider_goal.yaw - slider_goal_prev.yaw
            )
            if slider_delta > 1e-6:
                goal_state = GoalState(pos=slider_goal.pos.copy(), yaw=slider_goal.yaw)
            slider_goal_prev = slider_goal

            _apply_keyboard_goal_delta(
                keys, goal_state, step_xyz=args.key_step_xyz, step_yaw=args.key_step_yaw
            )

            goal_changed = (
                np.linalg.norm(goal_state.pos - last_applied_goal.pos) > 1e-9
                or abs(goal_state.yaw - last_applied_goal.yaw) > 1e-9
            )
            if goal_changed:
                _goal_to_scene(scene, goal_state)
                scene.sync_to_sim(sim)
                last_applied_goal = GoalState(pos=goal_state.pos.copy(), yaw=goal_state.yaw)

            x_goal = scene.get_goal_T()[:3, 3].copy()
            q_goal = pose_to_quat(scene.get_goal_T())[1]
            if np.dot(q_goal, last_goal_quat) < 0.0:
                q_goal *= -1.0
            _update_goal_marker(sim, marker_bid, x_goal)

            q_now = sim.get_joint_angles()
            T_now, _ = sim.fk(q_now)
            x_now, q_now_wxyz = pose_to_quat(T_now)

            now = time.perf_counter()
            goal_shift = np.linalg.norm(x_goal - last_goal_pos)
            goal_rot_shift = np.linalg.norm(q_goal - last_goal_quat)
            need_replan = (
                len(cached_plan_pos) < 2
                or plan_idx >= (len(cached_plan_pos) - 1)
                or (now - last_adapt_time) > (1.0 / max(args.adapt_hz, 0.1))
                or goal_shift > args.goal_replan_pos
                or goal_rot_shift > args.goal_replan_quat
            )

            if need_replan:
                t0 = time.perf_counter()
                try:
                    adapted = adapt_emp(
                        model=model,
                        new_x_start=x_now,
                        new_x_star=x_goal,
                        new_q_start=q_now_wxyz,
                        new_q_star=q_goal,
                        N_vel=args.n_vel,
                        eps_stab=args.eps_stab,
                    )
                    cached_plan_pos, cached_plan_quat = rollout_se3(
                        x_now,
                        q_now_wxyz,
                        adapted,
                        max_steps=args.plan_steps,
                        pos_tol=args.pos_tol,
                        ori_tol=args.ori_tol,
                    )
                    plan_idx = 1 if len(cached_plan_pos) > 1 else 0
                    last_adapt_time = now
                    last_goal_pos = x_goal.copy()
                    last_goal_quat = q_goal.copy()
                except Exception as exc:
                    print(f"[WARN] Replan failed: {exc}")
                last_adapt_ms = (time.perf_counter() - t0) * 1000.0

            if len(cached_plan_pos) > 0:
                i = min(plan_idx, len(cached_plan_pos) - 1)
                T_des = make_pose(cached_plan_pos[i], quat_wxyz=cached_plan_quat[i])
                q_des, _ok, _err = sim.ik(T_des, q0=q_now, tol=5e-3)
                sim.set_joint_angles(q_des, control=True)
                plan_idx = min(plan_idx + args.plan_stride, max(len(cached_plan_pos) - 1, 0))

            sim.step(args.steps_per_tick)
            T_actual, _ = sim.fk(sim.get_joint_angles())
            scene.update_held_objects(sim, T_actual)

            if (now - last_report_time) > 1.0:
                pos_err_cm = np.linalg.norm(T_actual[:3, 3] - x_goal) * 100.0
                print(
                    f"adapt {last_adapt_ms:6.1f} ms | "
                    f"goal [{x_goal[0]:.3f}, {x_goal[1]:.3f}, {x_goal[2]:.3f}] | "
                    f"ee->goal {pos_err_cm:5.1f} cm"
                )
                last_report_time = now

            sleep_s = loop_dt - (time.perf_counter() - t_loop)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    print("Exited realtime EMP loop.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realtime EMP adaptation loop for book insertion")
    p.add_argument("--loop-hz", type=float, default=50.0, help="Main control loop frequency")
    p.add_argument("--adapt-hz", type=float, default=3.0, help="Max adaptation rate")
    p.add_argument("--n-vel", type=int, default=80, help="Reference points for adaptation")
    p.add_argument("--eps-stab", type=float, default=1e-3, help="Stability margin")
    p.add_argument("--plan-steps", type=int, default=90, help="Rollout horizon per replan")
    p.add_argument("--plan-stride", type=int, default=2, help="Waypoint advance per control tick")
    p.add_argument("--steps-per-tick", type=int, default=3, help="PyBullet steps per control tick")
    p.add_argument("--pos-tol", type=float, default=0.008, help="Rollout positional tolerance")
    p.add_argument("--ori-tol", type=float, default=0.06, help="Rollout orientation tolerance")
    p.add_argument("--key-step-xyz", type=float, default=0.003, help="Keyboard xyz step size")
    p.add_argument("--key-step-yaw", type=float, default=np.radians(1.5), help="Keyboard yaw step")
    p.add_argument("--goal-replan-pos", type=float, default=0.004, help="Goal shift to force replan")
    p.add_argument("--goal-replan-quat", type=float, default=0.015, help="Goal quat shift to force replan")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
