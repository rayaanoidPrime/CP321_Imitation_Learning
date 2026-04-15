"""
run_book_rack_interactive_headless.py - Simulated interactive EMP adaptation with GIF recording.

Since we can't capture real keyboard input in headless mode, this script simulates
an interactive session where the goal moves along a scripted path (as if a user were
moving it with keyboard controls). The same real-time EMP adaptation loop is used.

The goal follows a path that mimics typical keyboard interactions:
1. Move right
2. Move forward  
3. Rotate
4. Move up/down
5. Complex combined movements

This produces the same visual result as manual keyboard control.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as pb

from emp_policy import adapt_emp, rollout_se3
from pb_stage1_env import PandaSimEnv, make_pose, pose_to_quat
from run_book_rack import step1_demos, step2_train
from task_book_rack import BookInsertScene


OUT_DIR = Path(os.environ.get("OUT_DIR", ".")) / "pybullet_replays"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GoalState:
    pos: np.ndarray
    yaw: float


def _capture_frame(sim: PandaSimEnv, width: int = 800, height: int = 600) -> np.ndarray:
    view = pb.computeViewMatrix(
        cameraEyePosition=[0.95, -0.35, 0.65],
        cameraTargetPosition=[0.50, 0.0, 0.18],
        cameraUpVector=[0, 0, 1],
        physicsClientId=sim._pcid,
    )
    proj = pb.computeProjectionMatrixFOV(
        fov=42, aspect=width / height, nearVal=0.05, farVal=10.0,
        physicsClientId=sim._pcid,
    )
    _, _, rgba, _, _ = pb.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER,
        physicsClientId=sim._pcid,
    )
    return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))


def _annotate(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame[..., :3])
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((12, 12, 620, 52), radius=8, fill=(15, 15, 15, 200))
    draw.text((24, 22), text, fill=(240, 240, 240))
    return img


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


def _goal_to_scene(scene: BookInsertScene, state: GoalState) -> None:
    scene.set_variant(
        rack_pos=state.pos,
        rack_euler=np.array([0.0, 0.0, state.yaw], dtype=float),
    )


def _make_keyboard_path(rack_pos0: np.ndarray, yaw0: float,
                        total_steps: int = 600) -> list[GoalState]:
    """
    Generate a goal trajectory that mimics keyboard-controlled movement.
    
    The path simulates a user pressing arrow keys, U/J, Q/E to move the rack
    around in various patterns - just like they would in the interactive version.
    """
    path = []
    pos = rack_pos0.copy()
    yaw = yaw0
    
    # Phase 1: Hold position (initial settling) - 30 steps
    for _ in range(30):
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 2: Move right (positive y) - 60 steps
    for _ in range(60):
        pos[1] += 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 3: Move forward (positive x) - 50 steps
    for _ in range(50):
        pos[0] += 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 4: Rotate counter-clockwise - 50 steps
    for _ in range(50):
        yaw += 0.01
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 5: Move up (positive z) - 30 steps
    for _ in range(30):
        pos[2] += 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 6: Move left (negative y) - 60 steps
    for _ in range(60):
        pos[1] -= 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 7: Rotate clockwise - 50 steps
    for _ in range(50):
        yaw -= 0.01
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 8: Move back (negative x) - 40 steps
    for _ in range(40):
        pos[0] -= 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 9: Move down (negative z) - 30 steps
    for _ in range(30):
        pos[2] -= 0.002
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 10: Complex diagonal movement - 60 steps
    for _ in range(60):
        pos[0] += 0.001
        pos[1] += 0.0015
        yaw += 0.005
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 11: Hold and let robot catch up - 50 steps
    for _ in range(50):
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Phase 12: Return toward original position - 100 steps
    target_pos = rack_pos0.copy()
    target_yaw = yaw0
    for i in range(100):
        alpha = (i + 1) / 100.0
        pos = (1 - alpha) * pos + alpha * target_pos
        yaw = (1 - alpha) * yaw + alpha * target_yaw
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    # Pad to total_steps if needed
    while len(path) < total_steps:
        path.append(GoalState(pos=pos.copy(), yaw=yaw))
    
    return path[:total_steps]


def run_interactive_simulation(args_dict: dict) -> str:
    """Run the interactive simulation and return the GIF path."""
    frames: list[Image.Image] = []
    
    with PandaSimEnv(gui=False) as sim:
        demos, scene = step1_demos(sim=sim)
        model = step2_train(demos)

        scene.spawn_in_sim(sim)
        q_start = scene.get_start_q().copy()
        sim.set_joint_angles(q_start)
        sim.step(3)

        rack_pos0 = scene.T_rack[:3, 3].copy()
        yaw0 = float(np.arctan2(scene.T_rack[1, 0], scene.T_rack[0, 0]))
        goal_state = GoalState(pos=rack_pos0.copy(), yaw=yaw0)

        marker_bid = _spawn_goal_marker(sim, scene.get_goal_T()[:3, 3])

        # Generate keyboard-like movement path
        keyboard_path = _make_keyboard_path(rack_pos0, yaw0, total_steps=600)
        path_idx = 0

        cached_plan_pos = np.zeros((0, 3), dtype=float)
        cached_plan_quat = np.zeros((0, 4), dtype=float)
        plan_idx = 0

        last_goal_pos = scene.get_goal_T()[:3, 3].copy()
        last_goal_quat = pose_to_quat(scene.get_goal_T())[1]
        last_applied_goal = GoalState(pos=goal_state.pos.copy(), yaw=goal_state.yaw)
        last_adapt_time = 0.0
        last_adapt_ms = 0.0
        
        loop_hz = args_dict.get("loop_hz", 50.0)
        adapt_hz = args_dict.get("adapt_hz", 3.0)
        n_vel = args_dict.get("n_vel", 80)
        eps_stab = args_dict.get("eps_stab", 1e-3)
        plan_steps = args_dict.get("plan_steps", 90)
        plan_stride = args_dict.get("plan_stride", 2)
        steps_per_tick = args_dict.get("steps_per_tick", 3)
        pos_tol = args_dict.get("pos_tol", 0.008)
        ori_tol = args_dict.get("ori_tol", 0.06)
        goal_replan_pos = args_dict.get("goal_replan_pos", 0.004)
        goal_replan_quat = args_dict.get("goal_replan_quat", 0.015)
        
        loop_dt = 1.0 / max(loop_hz, 1.0)
        frame_counter = 0

        print("\nSimulating interactive keyboard control...")
        print(f"Path length: {len(keyboard_path)} steps")
        
        for step in range(800):
            # Update goal from keyboard path
            if path_idx < len(keyboard_path):
                goal_state = keyboard_path[path_idx]
                path_idx += 1

            # Apply goal to scene
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
                or (now - last_adapt_time) > (1.0 / max(adapt_hz, 0.1))
                or goal_shift > goal_replan_pos
                or goal_rot_shift > goal_replan_quat
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
                        N_vel=n_vel,
                        eps_stab=eps_stab,
                    )
                    cached_plan_pos, cached_plan_quat = rollout_se3(
                        x_now,
                        q_now_wxyz,
                        adapted,
                        max_steps=plan_steps,
                        pos_tol=pos_tol,
                        ori_tol=ori_tol,
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
                plan_idx = min(plan_idx + plan_stride, max(len(cached_plan_pos) - 1, 0))

            sim.step(steps_per_tick)
            T_actual, _ = sim.fk(sim.get_joint_angles())
            scene.update_held_objects(sim, T_actual)

            # Capture every 3rd frame
            frame_counter += 1
            if frame_counter % 3 == 0:
                pos_err_cm = np.linalg.norm(T_actual[:3, 3] - x_goal) * 100.0
                label = (
                    f"Interactive Book Insert | "
                    f"goal=[{x_goal[0]:.2f}, {x_goal[1]:.2f}, {x_goal[2]:.2f}] | "
                    f"err={pos_err_cm:.1f}cm"
                )
                raw = _capture_frame(sim)
                frames.append(_annotate(raw, label))

            if step % 50 == 0:
                pos_err_cm = np.linalg.norm(T_actual[:3, 3] - x_goal) * 100.0
                print(
                    f"  step {step:3d} | "
                    f"adapt {last_adapt_ms:6.1f} ms | "
                    f"goal [{x_goal[0]:.3f}, {x_goal[1]:.3f}, {x_goal[2]:.3f}] | "
                    f"ee->goal {pos_err_cm:5.1f} cm | frames: {len(frames)}"
                )

            # Small sleep to prevent CPU hogging
            sleep_s = loop_dt - (time.perf_counter() - now)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    # Save GIF
    if frames:
        out_path = OUT_DIR / "book_insert_interactive_pybullet.gif"
        print(f"\nSaving {len(frames)} frames to {out_path} ...")
        frames[0].save(
            str(out_path),
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
            optimize=False,
        )
        print(f"Saved: {out_path}")
        return str(out_path)
    else:
        print("No frames captured!")
        return ""


def main() -> None:
    args = {
        "loop_hz": 50.0,
        "adapt_hz": 3.0,
        "n_vel": 80,
        "eps_stab": 1e-3,
        "plan_steps": 90,
        "plan_stride": 2,
        "steps_per_tick": 3,
        "pos_tol": 0.008,
        "ori_tol": 0.06,
        "goal_replan_pos": 0.004,
        "goal_replan_quat": 0.015,
    }
    run_interactive_simulation(args)


if __name__ == "__main__":
    main()
