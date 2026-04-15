"""
run_multi_pick_place.py — End-to-end EMP pipeline for the multi-step pick-and-place task.
"""

import os
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from task_multi_pick_place import (
    MultiPickPlaceScene,
    MULTI_PICK_PLACE_TASK,
    multi_pick_place_jitter,
    make_ood_scenes,
)
from emp_task_spec import generate_demonstrations, save_demos, load_demos
from emp_policy import (
    preprocess_demo,
    train_emp,
    adapt_emp,
    rollout_se3,
    stability_check,
    lyapunov_violation,
    plot_rollouts,
    execute_policy_in_sim,
)
from pb_stage1_env import PandaSimEnv, ik, make_pose, pose_to_quat, Q_NEUTRAL

OUT = os.environ.get("OUT_DIR", ".")
GUI = os.environ.get("GUI", "1") == "1"
SEED = 42
N_DEMOS = 7

os.makedirs(OUT, exist_ok=True)


def _section(title: str) -> str:
    return f"\n== {title} =="


def _demo_is_valid(demo: dict) -> bool:
    pos = np.asarray(demo.get("pos", []), dtype=float)
    quat = np.asarray(demo.get("quat", []), dtype=float)
    if len(pos) < 2 or len(quat) != len(pos):
        return False
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        return False
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return float(np.max(step)) < 0.15


def _scene_from_demo_goal(scene: MultiPickPlaceScene, demo: dict) -> None:
    T_goal = np.asarray(demo.get("T_goal"))
    if T_goal.shape != (4, 4):
        return
    place_2_pos = T_goal[:3, 3].copy()
    scene.set_variant(place_2_pos=place_2_pos)


def step1_demos(sim: PandaSimEnv = None) -> tuple:
    demo_path = os.path.join(OUT, "demos_multi_pick_place.npz")
    scene = MultiPickPlaceScene()

    demos = []
    if os.path.isfile(demo_path):
        print(f"Loading cached demos from {demo_path} ...")
        demos = load_demos(demo_path)
        if demos and all(_demo_is_valid(d) for d in demos):
            _scene_from_demo_goal(scene, demos[0])
            print(f"  Loaded {len(demos)} demos.")
        else:
            print("  Cached demos are invalid; regenerating.")
            demos = []

    if not demos:
        print(_section("Step 1: Generating multi-step pick-and-place demonstrations"))
        if sim is not None:
            scene.spawn_in_sim(sim)

        demos = generate_demonstrations(
            task_spec=MULTI_PICK_PLACE_TASK,
            scene=scene,
            N_demos=N_DEMOS,
            seed=SEED,
            sim=sim,
            scene_variant_fn=multi_pick_place_jitter,
        )
        save_demos(demos, demo_path)

    return demos, scene


def step2_train(demos: list) -> dict:
    print(_section("Step 2: Training EMP for multi-step pick-and-place"))
    t0 = time.perf_counter()
    data = preprocess_demo(demos[0], N_max=200)

    print(f"  Training data: N={len(data['X'])}")
    print(f"  x* = {np.round(data['x_star'], 3)}")
    print(f"  q* = {np.round(data['q_star'], 4)}")

    model = train_emp(data)
    elapsed = time.perf_counter() - t0

    ns, nt = stability_check(model["Ak_pos"], model["P"])
    ns_o = sum(1 for A in model["Ak_ori"]
               if np.linalg.eigvals(A + A.T).real.max() < 0)
    violation = lyapunov_violation(data["X"], data["Xdot"], model["P"], model["x_star"])
    print(f"  K={model['K']}   training time: {elapsed:.1f} s")
    print(f"  Position GAS:    {ns}/{nt}")
    print(f"  Orientation stable: {ns_o}/{model['K']}")
    print(f"  Lyapunov violation: {violation:.1%}")
    return model


def step3_adapt(model: dict) -> dict:
    print(_section("Step 3: Adapting to OOD scenes"))
    ood_scenes = make_ood_scenes()
    adapted_policies = {}

    for label, ood_scene in ood_scenes.items():
        t0 = time.perf_counter()
        T_start_ood = ood_scene.get_start_T()
        T_goal_ood = ood_scene.get_goal_T()

        new_x_start = T_start_ood[:3, 3]
        new_x_star = T_goal_ood[:3, 3]
        new_q_start = pose_to_quat(T_start_ood)[1]
        new_q_star = pose_to_quat(T_goal_ood)[1]

        pol = adapt_emp(model, new_x_start, new_x_star, new_q_start, new_q_star)
        traj_a = rollout_se3(new_x_start, new_q_start, pol)
        pos_a, quat_a = traj_a

        ns_a, nt_a = stability_check(pol["Ak_pos"], pol["P"])
        pos_err_cm = np.linalg.norm(pos_a[-1] - pol["x_star"]) * 100.0
        ori_err = np.linalg.norm(quat_a[-1] - pol["q_star"])
        elapsed = time.perf_counter() - t0
        print(
            f"  {label:<15} adapt: {elapsed:.1f}s  GAS: {ns_a}/{nt_a}  "
            f"final dist: {pos_err_cm:.1f} cm  quat err: {ori_err:.4f}"
        )

        adapted_policies[label] = (pol, traj_a)

    return adapted_policies


def step4_rollout(model: dict, demos: list) -> tuple:
    print(_section("Step 4: Original EMP rollout"))
    data = preprocess_demo(demos[0])
    x0 = data["X"][0]
    q0 = data["Q"][0]
    traj_orig = rollout_se3(x0, q0, model)
    pos_o, quat_o = traj_orig
    print(
        f"  Steps: {len(pos_o)}   "
        f"final dist to x*: {np.linalg.norm(pos_o[-1] - model['x_star']) * 100:.1f} cm   "
        f"quat err: {np.linalg.norm(quat_o[-1] - model['q_star']):.4f}"
    )
    return traj_orig


def step5_plots(model, traj_orig, adapted_policies, demos):
    print(_section("Step 5: Saving plots"))
    plot_rollouts(
        model=model,
        traj_orig=traj_orig,
        adapted_policies=adapted_policies,
        demo=demos[0],
        save_path=os.path.join(OUT, "multi_pick_place_rollouts.png"),
    )


def step6_visualise(demos, model, adapted_policies, scene: MultiPickPlaceScene):
    print(_section("Step 6: PyBullet visualisation"))

    with PandaSimEnv(gui=True) as sim:
        scene.spawn_in_sim(sim)
        time.sleep(0.5)

        print("  Playing demo trajectory ...")
        q_cur = demos[0]["q_joints"][0].copy()
        sim.set_joint_angles(q_cur)
        time.sleep(0.5)

        for q in demos[0]["q_joints"]:
            sim.set_joint_angles(q)
            sim.step()
            T_ee, _ = sim.fk(q)
            scene.update_held_objects(sim, T_ee)
            time.sleep(0.04)

        time.sleep(1.5)

        print("  Playing original EMP rollout ...")
        pos_o, quat_o = rollout_se3(demos[0]["pos"][0], demos[0]["quat"][0], model)
        q_cur = demos[0]["q_joints"][0].copy()
        sim.set_joint_angles(q_cur)
        time.sleep(0.5)

        for i in range(len(pos_o)):
            T = make_pose(pos_o[i], quat_wxyz=quat_o[i])
            q_cur, _ok, _ = ik(T, q0=q_cur)
            sim.set_joint_angles(q_cur)
            sim.step()
            T_ee, _ = sim.fk(q_cur)
            scene.update_held_objects(sim, T_ee)
            time.sleep(0.04)

        time.sleep(1.5)

        ood_label = list(adapted_policies.keys())[0]
        _pol, traj = adapted_policies[ood_label]
        pos_a, quat_a = traj
        ood_scene = make_ood_scenes()[ood_label]
        ood_scene.spawn_in_sim(sim)
        time.sleep(0.5)

        print(f"  Playing adapted rollout ({ood_label}) ...")
        q_cur = ood_scene.get_start_q().copy()

        for i in range(len(pos_a)):
            T = make_pose(pos_a[i], quat_wxyz=quat_a[i])
            q_cur, _ok, _ = ik(T, q0=q_cur)
            sim.set_joint_angles(q_cur)
            sim.step()
            T_ee, _ = sim.fk(q_cur)
            ood_scene.update_held_objects(sim, T_ee)
            time.sleep(0.04)

        print("  Done. Close the PyBullet window or press Enter to exit.")
        try:
            input()
        except EOFError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Multi-step pick-and-place EMP pipeline")
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip PyBullet GUI visualisation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plot output")
    args = parser.parse_args()

    show_gui = GUI and not args.no_gui
    show_plot = not args.no_plot

    t_total = time.perf_counter()

    with PandaSimEnv(gui=False) as sim:
        demos, scene = step1_demos(sim=sim)
        model = step2_train(demos)
        adapted = step3_adapt(model)
        traj_orig = step4_rollout(model, demos)

    if show_plot:
        step5_plots(model, traj_orig, adapted, demos)

    if show_gui:
        step6_visualise(demos, model, adapted, scene)

    print(f"\nTotal wall time: {time.perf_counter() - t_total:.1f} s")
    print("[OK] Done")


if __name__ == "__main__":
    main()
