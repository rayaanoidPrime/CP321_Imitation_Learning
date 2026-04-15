"""
make_pybullet_replays.py - Render PyBullet Franka Panda replay GIFs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as pb

from emp_policy import (
    preprocess_demo, train_emp, adapt_emp, rollout_se3,
    rollout_se3_with_obstacles,
)
from emp_task_spec import load_demos
from pb_stage1_env import PandaSimEnv, ik, make_pose, pose_to_quat, Q_NEUTRAL
from robomimic_loader import load_robomimic_task_demos
from run_robomimic_emp import TASKS as ROBOMIMIC_TASKS, make_adaptations
from task_book_rack import BookInsertScene, make_ood_scenes, book_rack_jitter
from task_cube_pouring import (
    CubePourScene, CUBE_POUR_TASK, cube_pan_jitter,
    make_ood_scenes as make_pour_ood_scenes,
)
from task_multi_pick_place import (
    MultiPickPlaceScene, MULTI_PICK_PLACE_TASK, multi_pick_place_jitter,
    make_ood_scenes as make_mpp_ood_scenes,
)
from task_obstacle_avoidance import (
    ObstacleAvoidanceScene, OBSTACLE_AVOIDANCE_TASK, obstacle_scene_jitter,
    make_ood_scenes as make_obs_ood_scenes,
)
from emp_task_spec import generate_demonstrations, save_demos


OUT_DIR = Path(os.environ.get("OUT_DIR", ".")) / "pybullet_replays"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Camera presets ────────────────────────────────────────────────────────────

CAM_DEFAULT = dict(
    eye=[1.2, -0.6, 0.85],
    target=[0.45, 0.0, 0.20],
    up=[0, 0, 1],
    fov=50,
)

CAM_CLOSEUP = dict(
    eye=[0.95, -0.35, 0.65],
    target=[0.50, 0.0, 0.18],
    up=[0, 0, 1],
    fov=42,
)

CAM_ROBOMIMIC = dict(
    eye=[1.45, -1.05, 1.18],
    target=[0.15, 0.02, 0.58],
    up=[0, 0, 1],
    fov=52,
)

CAM_WIDE = dict(
    eye=[1.5, -0.9, 1.0],
    target=[0.45, 0.0, 0.15],
    up=[0, 0, 1],
    fov=55,
)


def _capture_frame(sim: PandaSimEnv, cam: dict = CAM_DEFAULT,
                   width: int = 800, height: int = 600) -> np.ndarray:
    view = pb.computeViewMatrix(
        cameraEyePosition=cam["eye"],
        cameraTargetPosition=cam["target"],
        cameraUpVector=cam["up"],
        physicsClientId=sim._pcid,
    )
    proj = pb.computeProjectionMatrixFOV(
        fov=cam["fov"], aspect=width / height, nearVal=0.05, farVal=10.0,
        physicsClientId=sim._pcid,
    )
    _, _, rgba, _, _ = pb.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER,
        physicsClientId=sim._pcid,
    )
    return np.array(rgba, dtype=np.uint8).reshape((height, width, 4))


def _annotate(frame: np.ndarray, title: str) -> Image.Image:
    img = Image.fromarray(frame[..., :3])
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((12, 12, 560, 52), radius=8, fill=(15, 15, 15, 200))
    draw.text((24, 22), title, fill=(240, 240, 240))
    return img


def _save_gif(frames: list[Image.Image], path: Path, fps: int = 16) -> None:
    if not frames:
        raise RuntimeError(f"No frames rendered for {path}")
    duration = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def _clear_scene(sim: PandaSimEnv) -> None:
    """Safely remove all scene bodies, ignoring errors for already-removed bodies."""
    bidders = list(sim._scene_bodies)
    sim._scene_bodies.clear()
    for bid in bidders:
        try:
            pb.removeBody(bid, physicsClientId=sim._pcid)
        except pb.error:
            pass
        except Exception:
            pass
    for bid in bidders:
        try:
            pb.removeBody(bid, physicsClientId=sim._pcid)
        except Exception:
            pass


def _replay_segment(sim: PandaSimEnv, pos: np.ndarray, quat: np.ndarray,
                    title: str, frames: list[Image.Image],
                    update_scene=None, step_stride: int = 2,
                    camera: dict = CAM_DEFAULT, max_frames: int = 35) -> None:
    q_cur = sim.get_joint_angles().copy()
    total = len(pos)
    effective_stride = max(step_stride, total // max_frames) if total > max_frames else step_stride
    frame_count = 0
    for i in range(total):
        T = make_pose(pos[i], quat_wxyz=quat[i])
        q_cur, _ok, _err = ik(T, q0=q_cur, tol=5e-3)
        sim.set_joint_angles(q_cur, control=False)
        sim.step(1)
        if update_scene is not None:
            T_ee, _ = sim.fk(q_cur)
            update_scene(sim, T_ee)
        if i % effective_stride == 0 or i == total - 1:
            frames.append(_annotate(_capture_frame(sim, cam=camera), title))
            frame_count += 1
            remaining_slots = max_frames - frame_count
            if remaining_slots > 0 and i < total - 1:
                effective_stride = max(effective_stride, (total - i) // remaining_slots)

    for _ in range(3):
        sim.step(1)
        if update_scene is not None:
            T_ee, _ = sim.fk(q_cur)
            update_scene(sim, T_ee)
        frames.append(_annotate(_capture_frame(sim, cam=camera), title))


# ── Book Insert ───────────────────────────────────────────────────────────────

def build_book_insert_gif() -> Path:
    demos = load_demos("demos_book_insert.npz")
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        # Demo
        scene = BookInsertScene()
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, demo["pos"], demo["quat"], "Book Insert: demo", frames,
            update_scene=scene.update_held_objects, step_stride=1,
            camera=CAM_CLOSEUP,
        )

        # Original EMP rollout
        _clear_scene(sim)
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, original[0], original[1], "Book Insert: EMP original", frames,
            update_scene=scene.update_held_objects, step_stride=2,
            camera=CAM_CLOSEUP,
        )

        # OOD adaptations
        for label, ood_scene in make_ood_scenes().items():
            _clear_scene(sim)
            ood_scene.spawn_in_sim(sim)
            T0 = ood_scene.get_start_T()
            Tg = ood_scene.get_goal_T()
            x0, xg = T0[:3, 3], Tg[:3, 3]
            q0, qg = pose_to_quat(T0)[1], pose_to_quat(Tg)[1]
            pol = adapt_emp(model, x0, xg, q0, qg)
            traj = rollout_se3(x0, q0, pol)
            _replay_segment(
                sim, traj[0], traj[1], f"Book Insert: adapt {label}", frames,
                update_scene=ood_scene.update_held_objects, step_stride=2,
                camera=CAM_CLOSEUP,
            )

    path = OUT_DIR / "book_insert_pybullet.gif"
    _save_gif(frames, path)
    return path


# ── Book Insert Realtime Adaptation ──────────────────────────────────────────

def build_book_insert_realtime_gif() -> Path:
    """
    Replay that shows the EMP policy adapting in real-time as the goal moves.
    The robot starts executing, then the goal rack position shifts mid-motion
    and the policy replans to the new goal.
    """
    demos = load_demos("demos_book_insert.npz")
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        scene = BookInsertScene()
        scene.spawn_in_sim(sim)

        q_start = scene.get_start_q().copy()
        sim.set_joint_angles(q_start)
        sim.step(3)

        x_now = scene.get_start_T()[:3, 3].copy()
        q_now_wxyz = pose_to_quat(scene.get_start_T())[1]

        x_goal = scene.get_goal_T()[:3, 3].copy()
        q_goal = pose_to_quat(scene.get_goal_T())[1]

        adapted = adapt_emp(model, x_now, x_goal, q_now_wxyz, q_goal)
        plan_pos, plan_quat = rollout_se3(x_now, q_now_wxyz, adapted, max_steps=120)

        plan_idx = 0
        goal_shift_count = 0
        goal_shifts = [40, 80]

        for step in range(200):
            q_now = sim.get_joint_angles()
            T_now, _ = sim.fk(q_now)
            x_now, q_now_wxyz = pose_to_quat(T_now)

            if step in goal_shifts and goal_shift_count < len(goal_shifts):
                goal_shift_count += 1
                dp = np.array([0.0, 0.06 * goal_shift_count, 0.0])
                new_rack = scene.T_rack[:3, 3] + dp
                scene.set_variant(rack_pos=new_rack)
                scene.sync_to_sim(sim)

                x_goal = scene.get_goal_T()[:3, 3].copy()
                q_goal = pose_to_quat(scene.get_goal_T())[1]

                adapted = adapt_emp(model, x_now, x_goal, q_now_wxyz, q_goal)
                plan_pos, plan_quat = rollout_se3(x_now, q_now_wxyz, adapted, max_steps=120)
                plan_idx = 0

            if len(plan_pos) > 0:
                i = min(plan_idx, len(plan_pos) - 1)
                T_des = make_pose(plan_pos[i], quat_wxyz=plan_quat[i])
                q_des, _ok, _err = sim.ik(T_des, q0=q_now, tol=5e-3)
                sim.set_joint_angles(q_des, control=True)
                plan_idx = min(plan_idx + 2, max(len(plan_pos) - 1, 0))

            sim.step(3)
            T_actual, _ = sim.fk(sim.get_joint_angles())
            scene.update_held_objects(sim, T_actual)

            if step % 3 == 0:
                label = "realtime"
                if goal_shift_count >= 2:
                    label = "realtime: goal shifted x2"
                elif goal_shift_count >= 1:
                    label = "realtime: goal shifted"
                frames.append(_annotate(
                    _capture_frame(sim, cam=CAM_CLOSEUP), label
                ))

    path = OUT_DIR / "book_insert_realtime_pybullet.gif"
    _save_gif(frames, path, fps=14)
    return path


# ── Cube Pour ─────────────────────────────────────────────────────────────────

def _ensure_demos(path: str, task_spec, scene, variant_fn) -> list:
    if os.path.isfile(path):
        return load_demos(path)
    demos = generate_demonstrations(
        task_spec=task_spec, scene=scene, N_demos=7,
        seed=42, scene_variant_fn=variant_fn,
    )
    save_demos(demos, path)
    return demos


def build_cube_pour_gif() -> Path:
    demos = _ensure_demos("demos_cube_pour.npz", CUBE_POUR_TASK,
                          CubePourScene(), cube_pan_jitter)
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        scene = CubePourScene()
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, demo["pos"], demo["quat"], "Cube Pour: demo", frames,
            update_scene=scene.update_held_objects, step_stride=1,
            camera=CAM_CLOSEUP,
        )

        _clear_scene(sim)
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, original[0], original[1], "Cube Pour: EMP original", frames,
            update_scene=scene.update_held_objects, step_stride=2,
            camera=CAM_CLOSEUP,
        )

        for label, ood_scene in make_pour_ood_scenes().items():
            _clear_scene(sim)
            ood_scene.spawn_in_sim(sim)
            T0 = ood_scene.get_start_T()
            Tg = ood_scene.get_goal_T()
            x0, xg = T0[:3, 3], Tg[:3, 3]
            q0, qg = pose_to_quat(T0)[1], pose_to_quat(Tg)[1]
            pol = adapt_emp(model, x0, xg, q0, qg)
            traj = rollout_se3(x0, q0, pol)
            _replay_segment(
                sim, traj[0], traj[1], f"Cube Pour: adapt {label}", frames,
                update_scene=ood_scene.update_held_objects, step_stride=2,
                camera=CAM_CLOSEUP,
            )

    path = OUT_DIR / "cube_pour_pybullet.gif"
    _save_gif(frames, path)
    return path


# ── Multi-Step Pick & Place ──────────────────────────────────────────────────

def build_multi_pick_place_gif() -> Path:
    demos = _ensure_demos("demos_multi_pick_place.npz", MULTI_PICK_PLACE_TASK,
                          MultiPickPlaceScene(), multi_pick_place_jitter)
    demo = demos[0]
    data = preprocess_demo(demo, N_max=200)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        scene = MultiPickPlaceScene()
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, demo["pos"], demo["quat"], "Multi Pick&Place: demo", frames,
            update_scene=scene.update_held_objects, step_stride=1,
            camera=CAM_WIDE,
        )

        _clear_scene(sim)
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, original[0], original[1], "Multi Pick&Place: EMP original", frames,
            update_scene=scene.update_held_objects, step_stride=2,
            camera=CAM_WIDE,
        )

        for label, ood_scene in make_mpp_ood_scenes().items():
            _clear_scene(sim)
            ood_scene.spawn_in_sim(sim)
            T0 = ood_scene.get_start_T()
            Tg = ood_scene.get_goal_T()
            x0, xg = T0[:3, 3], Tg[:3, 3]
            q0, qg = pose_to_quat(T0)[1], pose_to_quat(Tg)[1]
            pol = adapt_emp(model, x0, xg, q0, qg)
            traj = rollout_se3(x0, q0, pol)
            _replay_segment(
                sim, traj[0], traj[1], f"Multi Pick&Place: adapt {label}", frames,
                update_scene=ood_scene.update_held_objects, step_stride=2,
                camera=CAM_WIDE,
            )

    path = OUT_DIR / "multi_pick_place_pybullet.gif"
    _save_gif(frames, path)
    return path


# ── Obstacle Avoidance ───────────────────────────────────────────────────────

def build_obstacle_avoidance_gif() -> Path:
    demos = _ensure_demos("demos_obstacle_avoidance.npz",
                          OBSTACLE_AVOIDANCE_TASK,
                          ObstacleAvoidanceScene(), obstacle_scene_jitter)
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)

    obstacle_pos = demos[0].get("_obstacle_pos", None)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        scene = ObstacleAvoidanceScene()
        scene.spawn_in_sim(sim)
        obs_pos = scene.get_obstacle_pos()
        obs_rad = scene.get_obstacle_radius()

        _replay_segment(
            sim, demo["pos"], demo["quat"], "Obstacle Avoid: demo", frames,
            update_scene=scene.update_held_objects, step_stride=1,
            camera=CAM_CLOSEUP,
        )

        _clear_scene(sim)
        scene.spawn_in_sim(sim)
        traj_nom = rollout_se3(data["X"][0], data["Q"][0], model)
        _replay_segment(
            sim, traj_nom[0], traj_nom[1], "Obstacle Avoid: no modulation", frames,
            update_scene=scene.update_held_objects, step_stride=2,
            camera=CAM_CLOSEUP,
        )

        _clear_scene(sim)
        scene.spawn_in_sim(sim)
        traj_obs = rollout_se3_with_obstacles(
            data["X"][0], data["Q"][0], model,
            obstacle_pos=obs_pos, obstacle_radius=obs_rad,
            influence_radius=0.12, obstacle_gain=0.5,
        )
        _replay_segment(
            sim, traj_obs[0], traj_obs[1], "Obstacle Avoid: with modulation", frames,
            update_scene=scene.update_held_objects, step_stride=2,
            camera=CAM_CLOSEUP,
        )

        for label, ood_scene in make_obs_ood_scenes().items():
            _clear_scene(sim)
            ood_scene.spawn_in_sim(sim)
            T0 = ood_scene.get_start_T()
            Tg = ood_scene.get_goal_T()
            x0, xg = T0[:3, 3], Tg[:3, 3]
            q0, qg = pose_to_quat(T0)[1], pose_to_quat(Tg)[1]
            pol = adapt_emp(model, x0, xg, q0, qg)
            obs_p = ood_scene.get_obstacle_pos()
            obs_r = ood_scene.get_obstacle_radius()
            traj = rollout_se3_with_obstacles(
                x0, q0, pol,
                obstacle_pos=obs_p, obstacle_radius=obs_r,
                influence_radius=0.12, obstacle_gain=0.5,
            )
            _replay_segment(
                sim, traj[0], traj[1], f"Obstacle Avoid: adapt {label}", frames,
                update_scene=ood_scene.update_held_objects, step_stride=2,
                camera=CAM_CLOSEUP,
            )

    path = OUT_DIR / "obstacle_avoidance_pybullet.gif"
    _save_gif(frames, path)
    return path


# ── Robomimic tasks ──────────────────────────────────────────────────────────

def _task_props(sim: PandaSimEnv, task_name: str, demo: dict) -> None:
    sim.remove_scene_bodies()
    pos = np.asarray(demo["pos"], dtype=float)
    start = pos[0]
    goal = pos[-1]

    if task_name == "lift":
        sim.add_box(
            pos=np.array([goal[0], goal[1], goal[2] - 0.08]),
            half_extents=np.array([0.025, 0.025, 0.025]),
            rgba=(0.95, 0.55, 0.20, 1.0),
        )
        sim.add_box(
            pos=np.array([goal[0], goal[1], goal[2] + 0.02]),
            half_extents=np.array([0.03, 0.03, 0.005]),
            rgba=(0.20, 0.75, 0.30, 0.6),
        )
    elif task_name == "can":
        sim.add_box(
            pos=np.array([start[0] + 0.06, start[1] - 0.02, 0.86]),
            half_extents=np.array([0.025, 0.025, 0.08]),
            rgba=(0.80, 0.20, 0.20, 1.0),
        )
        sim.add_box(
            pos=np.array([goal[0], goal[1], 0.90]),
            half_extents=np.array([0.06, 0.06, 0.04]),
            rgba=(0.20, 0.50, 0.90, 0.35),
        )
    elif task_name == "square":
        sim.add_box(
            pos=np.array([start[0] + 0.03, start[1], 0.87]),
            half_extents=np.array([0.03, 0.03, 0.015]),
            rgba=(0.95, 0.65, 0.20, 1.0),
        )
        sim.add_box(
            pos=np.array([goal[0], goal[1], 0.90]),
            half_extents=np.array([0.015, 0.015, 0.08]),
            rgba=(0.20, 0.75, 0.30, 0.85),
        )


def _replay_segment_robomimic(sim: PandaSimEnv, pos: np.ndarray, quat: np.ndarray,
                              title: str, frames: list[Image.Image],
                              gripper_width: float | None = None,
                              step_stride: int = 2, max_frames: int = 35) -> None:
    q_cur = sim.get_joint_angles().copy()
    total = len(pos)
    effective_stride = max(step_stride, total // max_frames) if total > max_frames else step_stride
    frame_count = 0
    for i in range(total):
        T = make_pose(pos[i], quat_wxyz=quat[i])
        q_cur, _ok, _err = ik(T, q0=q_cur, tol=5e-3)
        sim.set_joint_angles(q_cur, control=False)
        if gripper_width is not None:
            sim.open_gripper(width=gripper_width)
        sim.step(1)
        if i % effective_stride == 0 or i == total - 1:
            frames.append(_annotate(_capture_frame(sim, cam=CAM_ROBOMIMIC), title))
            frame_count += 1
            remaining_slots = max_frames - frame_count
            if remaining_slots > 0 and i < total - 1:
                effective_stride = max(effective_stride, (total - i) // remaining_slots)

    for _ in range(6):
        frames.append(_annotate(_capture_frame(sim, cam=CAM_ROBOMIMIC), title))


def build_robomimic_gif(task_name: str) -> Path:
    demos = load_robomimic_task_demos(task_name, limit=10)
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)

    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        _task_props(sim, task_name, demo)
        grip_demo = float(np.clip(np.mean(demo.get("gripper_qpos", np.zeros((1, 2)))), 0.02, 0.08))
        _replay_segment_robomimic(
            sim, demo["pos"], demo["quat"], f"{task_name}: demo", frames,
            gripper_width=grip_demo, step_stride=1,
        )

        _clear_scene(sim)
        _task_props(sim, task_name, demo)
        _replay_segment_robomimic(
            sim, original[0], original[1], f"{task_name}: EMP original", frames,
            gripper_width=0.04, step_stride=2,
        )

        x0 = data["X"][0]
        xg = data["x_star"]
        q0 = data["Q"][0]
        qg = data["q_star"]
        for label, cfg in make_adaptations(x0, xg, q0, qg).items():
            pol = adapt_emp(model, cfg["x0"], cfg["xg"], cfg["q0"], cfg["qg"])
            traj = rollout_se3(cfg["x0"], cfg["q0"], pol)
            _clear_scene(sim)
            _task_props(sim, task_name, demo)
            target = cfg["xg"]
            sim.add_box(
                pos=np.array([target[0], target[1], target[2] - 0.01]),
                half_extents=np.array([0.02, 0.02, 0.01]),
                rgba=(0.95, 0.95, 0.20, 0.55),
            )
            _replay_segment_robomimic(
                sim, traj[0], traj[1], f"{task_name}: adapt {label}", frames,
                gripper_width=0.04, step_stride=2,
            )

    path = OUT_DIR / f"{task_name}_pybullet.gif"
    _save_gif(frames, path)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    created = [build_book_insert_gif()]
    created.append(build_book_insert_realtime_gif())
    for task_name in ROBOMIMIC_TASKS:
        created.append(build_robomimic_gif(task_name))
    created.append(build_cube_pour_gif())
    created.append(build_multi_pick_place_gif())
    created.append(build_obstacle_avoidance_gif())

    print("Created PyBullet replay GIFs:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
