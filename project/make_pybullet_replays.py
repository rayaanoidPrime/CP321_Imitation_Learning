"""
make_pybullet_replays.py - Render PyBullet Franka Panda replay GIFs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as pb

from emp_policy import preprocess_demo, train_emp, adapt_emp, rollout_se3
from emp_task_spec import load_demos
from pb_stage1_env import PandaSimEnv, ik, make_pose, pose_to_quat
from robomimic_loader import load_robomimic_task_demos
from run_robomimic_emp import TASKS as ROBOMIMIC_TASKS, make_adaptations
from task_book_rack import BookInsertScene, make_ood_scenes


OUT_DIR = Path(os.environ.get("OUT_DIR", ".")) / "pybullet_replays"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _annotate(frame: np.ndarray, title: str) -> Image.Image:
    img = Image.fromarray(frame[..., :3])
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((12, 12, 330, 54), radius=8, fill=(20, 20, 20))
    draw.text((24, 24), title, fill=(240, 240, 240))
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


def _replay_segment(sim: PandaSimEnv, pos: np.ndarray, quat: np.ndarray,
                    title: str, frames: list[Image.Image],
                    update_scene=None, gripper_width: float | None = None,
                    step_stride: int = 2) -> None:
    q_cur = sim.get_joint_angles().copy()
    for i in range(len(pos)):
        T = make_pose(pos[i], quat_wxyz=quat[i])
        q_cur, _ok, _err = ik(T, q0=q_cur, tol=5e-3)
        sim.set_joint_angles(q_cur, control=False)
        if gripper_width is not None:
            sim.open_gripper(width=gripper_width)
        sim.step(1)
        if update_scene is not None:
            T_ee, _ = sim.fk(q_cur)
            update_scene(sim, T_ee)
        if i % step_stride == 0 or i == len(pos) - 1:
            frames.append(_annotate(sim.capture_screenshot(720, 540), title))

    for _ in range(10):
        frames.append(_annotate(sim.capture_screenshot(720, 540), title))


def build_book_insert_gif() -> Path:
    demos = load_demos("demos_book_insert.npz")
    demo = demos[0]
    data = preprocess_demo(demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)

    scene = BookInsertScene()
    frames: list[Image.Image] = []
    with PandaSimEnv(gui=False) as sim:
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, demo["pos"], demo["quat"], "book_insert: demo", frames,
            update_scene=scene.update_held_objects, step_stride=1,
        )
        scene.spawn_in_sim(sim)
        _replay_segment(
            sim, original[0], original[1], "book_insert: EMP original", frames,
            update_scene=scene.update_held_objects, step_stride=2,
        )

        for label, ood_scene in make_ood_scenes().items():
            T0 = ood_scene.get_start_T()
            Tg = ood_scene.get_goal_T()
            x0 = T0[:3, 3]
            xg = Tg[:3, 3]
            q0 = pose_to_quat(T0)[1]
            qg = pose_to_quat(Tg)[1]
            pol = adapt_emp(model, x0, xg, q0, qg)
            traj = rollout_se3(x0, q0, pol)
            ood_scene.spawn_in_sim(sim)
            _replay_segment(
                sim, traj[0], traj[1], f"book_insert: adapt {label}", frames,
                update_scene=ood_scene.update_held_objects, step_stride=2,
            )

    path = OUT_DIR / "book_insert_pybullet.gif"
    _save_gif(frames, path)
    return path


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
        _replay_segment(
            sim, demo["pos"], demo["quat"], f"{task_name}: demo", frames,
            gripper_width=grip_demo, step_stride=1,
        )

        _task_props(sim, task_name, demo)
        _replay_segment(
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
            _task_props(sim, task_name, demo)
            target = cfg["xg"]
            sim.add_box(
                pos=np.array([target[0], target[1], target[2] - 0.01]),
                half_extents=np.array([0.02, 0.02, 0.01]),
                rgba=(0.95, 0.95, 0.20, 0.55),
            )
            _replay_segment(
                sim, traj[0], traj[1], f"{task_name}: adapt {label}", frames,
                gripper_width=0.04, step_stride=2,
            )

    path = OUT_DIR / f"{task_name}_pybullet.gif"
    _save_gif(frames, path)
    return path


def main():
    created = [build_book_insert_gif()]
    for task_name in ROBOMIMIC_TASKS:
        created.append(build_robomimic_gif(task_name))

    print("Created PyBullet replay GIFs:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
