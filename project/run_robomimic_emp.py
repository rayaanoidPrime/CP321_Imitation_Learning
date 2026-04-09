"""
run_robomimic_emp.py - Train and adapt EMP policies on robomimic tasks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from emp_policy import preprocess_demo, train_emp, adapt_emp, rollout_se3, plot_rollouts
from robomimic_loader import load_robomimic_task_demos


OUT_DIR = Path(os.environ.get("OUT_DIR", ".")) / "robomimic_emp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ("lift", "can", "square")


def _wxyz_to_rot(q_wxyz: np.ndarray) -> Rot:
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def _rot_to_wxyz(rot: Rot) -> np.ndarray:
    q_xyzw = rot.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def make_adaptations(x0: np.ndarray, xg: np.ndarray,
                     q0: np.ndarray, qg: np.ndarray) -> dict:
    yaw_p = Rot.from_euler("z", 15, degrees=True)
    yaw_n = Rot.from_euler("z", -20, degrees=True)
    return {
        "translate": dict(
            x0=x0 + np.array([0.04, -0.03, 0.00]),
            xg=xg + np.array([0.05, -0.02, 0.02]),
            q0=q0.copy(),
            qg=qg.copy(),
        ),
        "yaw_goal": dict(
            x0=x0.copy(),
            xg=xg.copy(),
            q0=_rot_to_wxyz(yaw_p * _wxyz_to_rot(q0)),
            qg=_rot_to_wxyz(yaw_p * _wxyz_to_rot(qg)),
        ),
        "full_ood": dict(
            x0=x0 + np.array([-0.03, 0.04, 0.01]),
            xg=xg + np.array([0.06, 0.03, 0.03]),
            q0=_rot_to_wxyz(yaw_n * _wxyz_to_rot(q0)),
            qg=_rot_to_wxyz(yaw_n * _wxyz_to_rot(qg)),
        ),
    }


def run_task(task_name: str, n_demos: int = 10) -> dict:
    demos = load_robomimic_task_demos(task_name, limit=n_demos)
    ref_demo = demos[0]

    np.savez_compressed(OUT_DIR / f"{task_name}_demos.npz", demos=np.array(demos, dtype=object))

    data = preprocess_demo(ref_demo, N_max=150)
    model = train_emp(data)
    traj_orig = rollout_se3(data["X"][0], data["Q"][0], model)

    x0 = data["X"][0]
    xg = data["x_star"]
    q0 = data["Q"][0]
    qg = data["q_star"]

    adapted = {}
    summary = {
        "task": task_name,
        "train_demo_len": int(len(ref_demo["pos"])),
        "num_loaded_demos": int(len(demos)),
        "K": int(model["K"]),
        "original_steps": int(len(traj_orig[0])),
        "original_final_dist_cm": float(np.linalg.norm(traj_orig[0][-1] - xg) * 100.0),
    }

    for label, cfg in make_adaptations(x0, xg, q0, qg).items():
        pol = adapt_emp(model, cfg["x0"], cfg["xg"], cfg["q0"], cfg["qg"])
        traj = rollout_se3(cfg["x0"], cfg["q0"], pol)
        adapted[label] = (pol, traj)
        summary[f"{label}_steps"] = int(len(traj[0]))
        summary[f"{label}_final_dist_cm"] = float(np.linalg.norm(traj[0][-1] - cfg["xg"]) * 100.0)
        summary[f"{label}_quat_err"] = float(np.linalg.norm(traj[1][-1] - cfg["qg"]))

    plot_rollouts(
        model=model,
        traj_orig=traj_orig,
        adapted_policies=adapted,
        demo=ref_demo,
        save_path=str(OUT_DIR / f"{task_name}_rollouts.png"),
    )
    return summary


def main():
    all_summary = {}
    for task_name in TASKS:
        print(f"\n== robomimic task: {task_name} ==")
        all_summary[task_name] = run_task(task_name)
        for key, value in all_summary[task_name].items():
            if key != "task":
                print(f"  {key}: {value}")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
