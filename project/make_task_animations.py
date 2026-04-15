"""
make_task_animations.py - Render offline GIF animations for EMP task rollouts.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from emp_policy import preprocess_demo, train_emp, adapt_emp, rollout_se3
from pb_utils import _set_axes_equal
from robomimic_loader import load_robomimic_task_demos
from run_robomimic_emp import TASKS as ROBOMIMIC_TASKS, make_adaptations
from emp_task_spec import load_demos
from task_book_rack import make_ood_scenes
from pb_stage1_env import pose_to_quat


OUT_DIR = Path(os.environ.get("OUT_DIR", ".")) / "animations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "demo": "#3B8BD4",
    "original": "#1D9E75",
    "translate": "#D85A30",
    "yaw_goal": "#9933CC",
    "full_ood": "#CC3366",
    "rotate": "#9933CC",
}


def _resample_path(pos: np.ndarray, n_frames: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    if len(pos) == 0:
        return pos
    if len(pos) == 1:
        return np.repeat(pos, n_frames, axis=0)
    t_src = np.linspace(0.0, 1.0, len(pos))
    t_dst = np.linspace(0.0, 1.0, n_frames)
    return np.column_stack([np.interp(t_dst, t_src, pos[:, d]) for d in range(3)])


def _make_animation(title: str, series: dict, save_path: Path, fps: int = 20) -> None:
    n_frames = max(90, max(len(v) for v in series.values()))
    sampled = {k: _resample_path(v, n_frames) for k, v in series.items()}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = np.concatenate(list(sampled.values()), axis=0)

    lines = {}
    points = {}
    for label in sampled:
        (line,) = ax.plot([], [], [], lw=2.2, color=COLORS.get(label, "#444444"), label=label)
        (point,) = ax.plot([], [], [], marker="o", ms=6, color=COLORS.get(label, "#444444"))
        lines[label] = line
        points[label] = point

    ax.scatter(all_pts[0, 0], all_pts[0, 1], all_pts[0, 2], s=25, color="black", marker="x")
    ax.scatter(all_pts[-1, 0], all_pts[-1, 1], all_pts[-1, 2], s=40, color="black", marker="*")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize=8)

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.15, 0.02)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    _set_axes_equal(ax)

    def update(frame_idx: int):
        for label, pts in sampled.items():
            seg = pts[: frame_idx + 1]
            lines[label].set_data(seg[:, 0], seg[:, 1])
            lines[label].set_3d_properties(seg[:, 2])
            points[label].set_data([seg[-1, 0]], [seg[-1, 1]])
            points[label].set_3d_properties([seg[-1, 2]])
        ax.view_init(elev=28, azim=35 + 0.25 * frame_idx)
        return list(lines.values()) + list(points.values())

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def build_book_insert_animation() -> Path:
    demos = load_demos("demos_book_insert.npz")
    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    series = {
        "demo": ref_demo["pos"],
        "original": original,
    }

    for label, scene in make_ood_scenes().items():
        T0 = scene.get_start_T()
        Tg = scene.get_goal_T()
        x0 = T0[:3, 3]
        xg = Tg[:3, 3]
        q0 = pose_to_quat(T0)[1]
        qg = pose_to_quat(Tg)[1]
        pol = adapt_emp(model, x0, xg, q0, qg)
        series[label] = rollout_se3(x0, q0, pol)[0]

    out = OUT_DIR / "book_insert.gif"
    _make_animation("Book Insert EMP", series, out)
    return out


def build_robomimic_animation(task_name: str) -> Path:
    demos = load_robomimic_task_demos(task_name, limit=10)
    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    series = {
        "demo": ref_demo["pos"],
        "original": original,
    }
    x0 = data["X"][0]
    xg = data["x_star"]
    q0 = data["Q"][0]
    qg = data["q_star"]

    for label, cfg in make_adaptations(x0, xg, q0, qg).items():
        pol = adapt_emp(model, cfg["x0"], cfg["xg"], cfg["q0"], cfg["qg"])
        series[label] = rollout_se3(cfg["x0"], cfg["q0"], pol)[0]

    out = OUT_DIR / f"robomimic_{task_name}.gif"
    _make_animation(f"robomimic {task_name} EMP", series, out)
    return out


def build_robomimic_animation(task_name: str) -> Path:
    demos = load_robomimic_task_demos(task_name, limit=10)
    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    series = {
        "demo": ref_demo["pos"],
        "original": original,
    }
    x0 = data["X"][0]
    xg = data["x_star"]
    q0 = data["Q"][0]
    qg = data["q_star"]

    for label, cfg in make_adaptations(x0, xg, q0, qg).items():
        pol = adapt_emp(model, cfg["x0"], cfg["xg"], cfg["q0"], cfg["qg"])
        series[label] = rollout_se3(cfg["x0"], cfg["q0"], pol)[0]

    out = OUT_DIR / f"robomimic_{task_name}.gif"
    _make_animation(f"robomimic {task_name} EMP", series, out)
    return out


def build_cube_pour_animation() -> Path:
    from task_cube_pouring import (
        CubePourScene, CUBE_POUR_TASK, cube_pan_jitter,
        make_ood_scenes as make_pour_ood_scenes,
    )
    from emp_task_spec import generate_demonstrations, save_demos

    demo_path = "demos_cube_pour.npz"
    scene = CubePourScene()
    demos = []
    if os.path.isfile(demo_path):
        demos = load_demos(demo_path)
    if not demos:
        demos = generate_demonstrations(
            task_spec=CUBE_POUR_TASK, scene=scene, N_demos=7,
            seed=42, scene_variant_fn=cube_pan_jitter,
        )
        save_demos(demos, demo_path)

    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    series = {"demo": ref_demo["pos"], "original": original}

    for label, ood_scene in make_pour_ood_scenes().items():
        T0 = ood_scene.get_start_T()
        Tg = ood_scene.get_goal_T()
        x0, q0 = T0[:3, 3], pose_to_quat(T0)[1]
        xg, qg = Tg[:3, 3], pose_to_quat(Tg)[1]
        pol = adapt_emp(model, x0, xg, q0, qg)
        series[label] = rollout_se3(x0, q0, pol)[0]

    out = OUT_DIR / "cube_pour.gif"
    _make_animation("Cube Pour EMP", series, out)
    return out


def build_multi_pick_place_animation() -> Path:
    from task_multi_pick_place import (
        MultiPickPlaceScene, MULTI_PICK_PLACE_TASK, multi_pick_place_jitter,
        make_ood_scenes as make_mpp_ood_scenes,
    )
    from emp_task_spec import generate_demonstrations, save_demos

    demo_path = "demos_multi_pick_place.npz"
    scene = MultiPickPlaceScene()
    demos = []
    if os.path.isfile(demo_path):
        demos = load_demos(demo_path)
    if not demos:
        demos = generate_demonstrations(
            task_spec=MULTI_PICK_PLACE_TASK, scene=scene, N_demos=7,
            seed=42, scene_variant_fn=multi_pick_place_jitter,
        )
        save_demos(demos, demo_path)

    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=200)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    series = {"demo": ref_demo["pos"], "original": original}

    for label, ood_scene in make_mpp_ood_scenes().items():
        T0 = ood_scene.get_start_T()
        Tg = ood_scene.get_goal_T()
        x0, q0 = T0[:3, 3], pose_to_quat(T0)[1]
        xg, qg = Tg[:3, 3], pose_to_quat(Tg)[1]
        pol = adapt_emp(model, x0, xg, q0, qg)
        series[label] = rollout_se3(x0, q0, pol)[0]

    out = OUT_DIR / "multi_pick_place.gif"
    _make_animation("Multi-Step Pick & Place EMP", series, out)
    return out


def build_obstacle_avoidance_animation() -> Path:
    from task_obstacle_avoidance import (
        ObstacleAvoidanceScene, OBSTACLE_AVOIDANCE_TASK, obstacle_scene_jitter,
        make_ood_scenes as make_obs_ood_scenes,
    )
    from emp_policy import rollout_se3_with_obstacles
    from emp_task_spec import generate_demonstrations, save_demos

    demo_path = "demos_obstacle_avoidance.npz"
    scene = ObstacleAvoidanceScene()
    demos = []
    if os.path.isfile(demo_path):
        demos = load_demos(demo_path)
    if not demos:
        demos = generate_demonstrations(
            task_spec=OBSTACLE_AVOIDANCE_TASK, scene=scene, N_demos=7,
            seed=42, scene_variant_fn=obstacle_scene_jitter,
        )
        save_demos(demos, demo_path)

    ref_demo = demos[0]
    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    original = rollout_se3(data["X"][0], data["Q"][0], model)[0]

    obstacle_pos = scene.get_obstacle_pos()
    obstacle_radius = scene.get_obstacle_radius()
    with_obstacle = rollout_se3_with_obstacles(
        data["X"][0], data["Q"][0], model,
        obstacle_pos=obstacle_pos, obstacle_radius=obstacle_radius,
        influence_radius=0.12, obstacle_gain=0.5,
    )[0]

    series = {"demo": ref_demo["pos"], "original": original, "with_obstacle": with_obstacle}

    for label, ood_scene in make_obs_ood_scenes().items():
        T0 = ood_scene.get_start_T()
        Tg = ood_scene.get_goal_T()
        x0, q0 = T0[:3, 3], pose_to_quat(T0)[1]
        xg, qg = Tg[:3, 3], pose_to_quat(Tg)[1]
        pol = adapt_emp(model, x0, xg, q0, qg)
        obs_pos = ood_scene.get_obstacle_pos()
        obs_rad = ood_scene.get_obstacle_radius()
        series[label] = rollout_se3_with_obstacles(
            x0, q0, pol,
            obstacle_pos=obs_pos, obstacle_radius=obs_rad,
            influence_radius=0.12, obstacle_gain=0.5,
        )[0]

    out = OUT_DIR / "obstacle_avoidance.gif"
    _make_animation("Obstacle Avoidance EMP", series, out)
    return out


def main():
    created = [build_book_insert_animation()]
    for task_name in ROBOMIMIC_TASKS:
        created.append(build_robomimic_animation(task_name))
    created.append(build_cube_pour_animation())
    created.append(build_multi_pick_place_animation())
    created.append(build_obstacle_avoidance_animation())

    print("Created animations:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
