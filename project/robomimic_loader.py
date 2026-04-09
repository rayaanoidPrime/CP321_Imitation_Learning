"""
robomimic_loader.py - Minimal utilities for loading robomimic low-dim demos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


TASK_FILES = {
    "lift": Path("data/robomimic/lift_low_dim_v15.hdf5"),
    "can": Path("data/robomimic/can_low_dim_v15.hdf5"),
    "square": Path("data/robomimic/square_low_dim_v15.hdf5"),
}


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


def _load_env_dt(h5f: h5py.File) -> float:
    env_args = json.loads(h5f["data"].attrs["env_args"])
    control_freq = float(env_args["env_kwargs"]["control_freq"])
    return 1.0 / control_freq


def _demo_to_emp_dict(demo_group: h5py.Group, dt: float) -> dict:
    obs = demo_group["obs"]
    pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
    quat = _normalise_quat_sequence(np.asarray(obs["robot0_eef_quat"], dtype=float))
    q_joints = np.asarray(obs["robot0_joint_pos"], dtype=float)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    obj = np.asarray(obs["object"], dtype=float)
    t = np.arange(len(pos), dtype=float) * dt

    vel = np.gradient(pos, axis=0) / dt
    ang_vel = np.zeros((len(quat), 3), dtype=float)

    return {
        "pos": pos,
        "quat": quat,
        "vel": vel,
        "ang_vel": ang_vel,
        "q_joints": q_joints,
        "gripper_qpos": gripper,
        "object": obj,
        "times": t,
        "T_goal": np.eye(4),
        "phase_names": ["trajectory"],
        "phase_starts": [0],
    }


def load_robomimic_task_demos(task_name: str, limit: int = 10) -> List[dict]:
    path = TASK_FILES[task_name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset for task '{task_name}': {path}")

    demos = []
    with h5py.File(path, "r") as f:
        dt = _load_env_dt(f)
        demo_keys = sorted(f["data"].keys())[:limit]
        for key in demo_keys:
            demos.append(_demo_to_emp_dict(f["data"][key], dt))
    return demos


def load_all_tasks(limit: int = 10) -> Dict[str, List[dict]]:
    return {task: load_robomimic_task_demos(task, limit=limit) for task in TASK_FILES}
