"""
pb_stage2_demos.py — Demonstration Trajectory Generation (PyBullet)
====================================================================
Drop-in replacement for stage2_demos.py.
Generates N_DEMOS kinematically-valid book-placing demonstrations.

Changes from original:
  • Imports from pb_stage1_env (PyBullet FK / IK)
  • Adds pb_render_demo()  — renders demo in PyBullet and saves video frames
  • Adds visualise_pb_demo() — saves a grid of PyBullet screenshots

All original functions (generate_demonstrations, save_demos, load_demos,
run_stage2_tests, plot_*) are preserved verbatim.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Drop-in replace: was "from stage1_env import ..."
from pb_stage1_env import (
    fk, ik, Q_NEUTRAL, JOINT_LIMITS,
    BookRackScene, make_pose, pose_to_quat,
    se3_interpolate, draw_arm, draw_frame,
    draw_box, visualise_scene, _set_axes_equal,
    _get_sim, PandaSimEnv,
)


# ══════════════════════════════════════════════════════════════════════════════
# 2.1  Task waypoint generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_task_waypoints(scene:       BookRackScene,
                             jitter_pos: float              = 0.03,
                             jitter_z:   float              = 0.02,
                             arc_height: float              = 0.12,
                             rng:        np.random.Generator = None):
    """
    Return [(T_waypoint, n_steps)] pairs for one demonstration.
    Phases:  A=LIFT  B=SWING  C=INSERT  D=RETRACT
    """
    if rng is None:
        rng = np.random.default_rng()

    T_slot     = scene.get_insert_pose()
    T_start, _ = scene.get_start_pose()

    # Jitter start
    dp     = rng.uniform(-jitter_pos, jitter_pos, size=3)
    dp[2]  = rng.uniform(-jitter_z, jitter_z)
    T_start_j = T_start.copy()
    T_start_j[:3, 3] += dp

    # Phase A: LIFT
    T_lift = T_start_j.copy()
    T_lift[:3, 3] += np.array([0.0, 0.0, arc_height + rng.uniform(-0.02, 0.02)])

    # Phase B: SWING (via-point above slot)
    T_via = scene.T_approach.copy()
    T_via[:3, 3] += np.array([
        rng.uniform(-0.02, 0.02),
        rng.uniform(-0.02, 0.02),
        arc_height + rng.uniform(-0.01, 0.02),
    ])
    T_via[:3, :3] = T_slot[:3, :3]

    # Phase C: INSERT
    T_insert = T_slot.copy()

    # Phase D: RETRACT
    retract_dist = 0.05 + rng.uniform(0, 0.02)
    T_retract = T_insert.copy()
    T_retract[:3, 3] += -scene.T_rack[:3, 0] * retract_dist

    return [
        (T_start_j, 1),
        (T_lift,   15),
        (T_via,    45),
        (T_insert, 30),
        (T_retract,10),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 2.2  Trajectory interpolation and IK solve
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_waypoints(waypoints, dt: float = 0.01):
    """
    SE(3) interpolate between waypoints and solve IK at each sample.

    Returns
    -------
    poses   : list of (4,4) EE transforms
    q_traj  : (T, 7) joint angle array
    times   : (T,) timestamps
    """
    poses  = []
    q_traj = []
    times  = []
    t      = 0.0
    q_cur  = Q_NEUTRAL.copy()

    for i in range(len(waypoints) - 1):
        T_a, _       = waypoints[i]
        T_b, n_steps = waypoints[i + 1]
        for k in range(n_steps):
            s    = k / max(n_steps - 1, 1)
            s_mj = s**3 * (10 - 15*s + 6*s**2)   # minimum-jerk profile
            T_k  = se3_interpolate(T_a, T_b, s_mj)
            q_k, ok, _err = ik(T_k, q0=q_cur, tol=5e-4)
            if ok:
                q_cur = q_k
            T_fk, _ = fk(q_cur)
            poses.append(T_fk)
            q_traj.append(q_cur.copy())
            times.append(t)
            t += dt

    return poses, np.array(q_traj), np.array(times)


def poses_to_arrays(poses, times):
    """SE(3) pose list → (pos, quat_wxyz, vel, ang_vel) arrays."""
    N   = len(poses)
    pos = np.array([T[:3, 3] for T in poses])
    rot = np.array([pose_to_quat(T)[1] for T in poses])  # wxyz

    dt  = np.diff(times, prepend=times[0] - 0.01)
    vel = np.gradient(pos, axis=0) / dt[:, None]

    ang_vel = np.zeros((N, 3))
    for i in range(1, N):
        dq   = rot[i] - rot[i-1]
        dt_i = float(times[i] - times[i-1]) if times[i] > times[i-1] else 0.01
        ang_vel[i] = 2.0 * dq[1:] / dt_i

    return pos, rot, vel, ang_vel


# ══════════════════════════════════════════════════════════════════════════════
# 2.3  Generate N_DEMOS demonstrations
# ══════════════════════════════════════════════════════════════════════════════

def generate_demonstrations(N_DEMOS: int = 7, seed: int = 42):
    """
    Generate N_DEMOS complete book-placing demonstrations.

    Returns
    -------
    demos : list of dicts  (pos, quat, vel, ang_vel, q_joints, times,
                            T_slot, T_rack)
    scene : BookRackScene  (last used scene)
    """
    rng   = np.random.default_rng(seed)
    scene = BookRackScene()
    demos = []

    print(f"Generating {N_DEMOS} demonstrations …")
    for i in range(N_DEMOS):
        rack_jitter    = rng.uniform(-0.01, 0.01, size=3)
        rack_jitter[2] = 0.0
        scene.set_rack_pose(
            np.array([0.55, -0.10, 0.15]) + rack_jitter)

        wpts = generate_task_waypoints(
            scene,
            jitter_pos = 0.025,
            jitter_z   = 0.015,
            arc_height = 0.12 + rng.uniform(-0.02, 0.03),
            rng        = rng,
        )
        poses, q_traj, times = interpolate_waypoints(wpts, dt=0.01)
        pos, quat, vel, ang_vel = poses_to_arrays(poses, times)

        # Small noise for human-like variability
        noise_pos  = rng.normal(0, 0.003, size=pos.shape)
        noise_q    = rng.normal(0, 0.005, size=quat.shape)
        pos_n      = pos + noise_pos
        quat_n     = quat + noise_q
        quat_n    /= np.linalg.norm(quat_n, axis=1, keepdims=True)

        demos.append(dict(
            pos      = pos_n,
            quat     = quat_n,
            vel      = vel,
            ang_vel  = ang_vel,
            q_joints = q_traj,
            times    = times,
            T_slot   = scene.get_insert_pose().copy(),
            T_rack   = scene.T_rack.copy(),
        ))
        print(f"  Demo {i+1:2d}: {len(times):4d} steps  "
              f"start={np.round(pos[0],3)}  end={np.round(pos[-1],3)}")

    return demos, scene


# ══════════════════════════════════════════════════════════════════════════════
# 2.4  Save / load
# ══════════════════════════════════════════════════════════════════════════════

def save_demos(demos, path: str = 'demos.npz') -> None:
    """Save all demonstrations to a compressed numpy archive."""
    arrays = {}
    for i, d in enumerate(demos):
        for k, v in d.items():
            arrays[f'demo{i}_{k}'] = np.array(v)
    arrays['n_demos'] = np.array(len(demos))
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(demos)} demos → {path}")


def load_demos(path: str = 'demos.npz'):
    """Load demonstrations from archive."""
    data  = np.load(path, allow_pickle=True)
    n     = int(data['n_demos'])
    keys  = ['pos','quat','vel','ang_vel','q_joints','times','T_slot','T_rack']
    return [{k: data[f'demo{i}_{k}'] for k in keys} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# 2.5  PyBullet rendering helper (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def pb_render_demo(demo: dict,
                   scene: BookRackScene,
                   out_dir: str    = ".",
                   demo_idx: int   = 0,
                   every_n: int    = 5,
                   width: int      = 480,
                   height: int     = 360) -> list:
    """
    Replay a demonstration in PyBullet and capture screenshots.

    Parameters
    ----------
    demo     : single demo dict from generate_demonstrations()
    scene    : BookRackScene to visualise
    out_dir  : directory to write PNG frames
    demo_idx : label for filenames
    every_n  : capture every N-th step
    width/height : image dimensions

    Returns list of frame file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    sim   = _get_sim()
    scene.spawn_in_sim(sim)

    q_traj = demo['q_joints']     # (T, 7)
    poses  = demo['pos']          # (T, 3)  EE positions

    frames = []
    for step_i, (q, ee_pos) in enumerate(zip(q_traj, poses)):
        sim.set_joint_angles(q)
        # Teleport book to EE position (gripper holding book)
        q_book = demo['quat'][step_i]        # wxyz quaternion at this timestep
        T_book = make_pose(ee_pos, quat_wxyz=q_book)
        scene.move_book_to(sim, T_book)

        if step_i % every_n == 0:
            img = sim.capture_screenshot(width, height)
            fpath = os.path.join(out_dir, f"demo{demo_idx}_frame{step_i:04d}.png")
            try:
                import PIL.Image
                PIL.Image.fromarray(img[:, :, :3]).save(fpath)
            except ImportError:
                # Fallback: save with matplotlib
                import matplotlib.pyplot as _mplt
                fig, ax = _mplt.subplots(1, 1, figsize=(width/100, height/100))
                ax.imshow(img[:, :, :3])
                ax.axis('off')
                fig.savefig(fpath, dpi=100, bbox_inches='tight', pad_inches=0)
                _mplt.close(fig)
            frames.append(fpath)

    print(f"  Saved {len(frames)} frames for demo {demo_idx}")
    return frames


def visualise_pb_demos(demos: list,
                       scene: BookRackScene,
                       save_path: str = "s2_pb_snapshots.png") -> None:
    """
    Capture 4 PyBullet screenshots per demo (key phases) and save as a grid.
    """
    sim = _get_sim()
    scene.spawn_in_sim(sim)

    n_demos = min(len(demos), 4)    # show at most 4 demos
    n_phases = 4
    fig, axes = plt.subplots(n_demos, n_phases,
                              figsize=(n_phases*3, n_demos*2.5))
    if n_demos == 1:
        axes = axes[np.newaxis, :]

    for row, d in enumerate(demos[:n_demos]):
        T_len = len(d['times'])
        idxs  = [0, T_len//5, T_len//2, T_len-1]
        lbls  = ['Start', 'Lift', 'Approach', 'Insert']

        for col, (idx, lbl) in enumerate(zip(idxs, lbls)):
            sim.set_joint_angles(d['q_joints'][idx])
            img = sim.capture_screenshot(320, 240)
            axes[row, col].imshow(img[:, :, :3])
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(lbl, fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(f'Demo {row+1}', fontsize=8)

    plt.suptitle('Stage 2 (PyBullet) — Key Phases for 4 Demos', fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.6  Stage 2 Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_stage2_tests(demos, scene) -> tuple:
    print("=" * 60)
    print("STAGE 2 TESTS — Demonstration Generation")
    print("=" * 60)
    passed = 0; total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        sym = "✓" if cond else "✗"
        print(f"  [{sym}] {name}" + (f"  ({detail})" if detail else ""))
        if cond:
            passed += 1
        return cond

    N = len(demos)
    check("Generated correct number of demos", N == 7, f"N={N}")

    for i, d in enumerate(demos):
        T  = len(d['times'])
        check(f"Demo {i+1}: ≥50 timesteps", T >= 50, f"T={T}")

        dp       = np.diff(d['pos'], axis=0)
        max_step = float(np.linalg.norm(dp, axis=1).max())
        check(f"Demo {i+1}: position continuity (max_step < 8 cm)",
              max_step < 0.08, f"max_step={max_step*100:.1f} cm")

        qnorms = np.linalg.norm(d['quat'], axis=1)
        check(f"Demo {i+1}: quaternions normalised",
              bool(np.allclose(qnorms, 1.0, atol=1e-4)),
              f"max_dev={np.abs(qnorms-1).max():.2e}")

        slot_pos = d['T_slot'][:3, 3]
        end_pos  = d['pos'][-1]
        dist_end = float(np.linalg.norm(end_pos - slot_pos))
        check(f"Demo {i+1}: EE ends near slot (< 8 cm)",
              dist_end < 0.08, f"dist={dist_end*100:.1f} cm")

    starts    = np.array([d['pos'][0] for d in demos])
    std_start = float(starts.std(axis=0).mean())
    check("Cross-demo start variability > 0.5 mm",
          std_start > 0.0005, f"std={std_start*1000:.1f} mm")

    print(f"\n  Result: {passed}/{total} tests passed")
    return passed, total


# ══════════════════════════════════════════════════════════════════════════════
# 2.7  Matplotlib visualisation (unchanged from stage2_demos.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_demo_overview(demos, scene,
                       save_path="s2_demos_3d.png") -> None:
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(demos)))
    draw_box(ax, make_pose(np.array([0.4, 0., -0.02])),
             np.array([0.9, 0.8, 0.04]), color='#C8AD7F', alpha=0.25)
    draw_box(ax, scene.T_rack,
             np.array([0.22, 0.32, 0.32]), color='#8B6914', alpha=0.20)
    draw_frame(ax, scene.T_slot_world, scale=0.06, label='slot')
    for i, (d, c) in enumerate(zip(demos, colours)):
        p = d['pos']
        ax.plot(p[:,0], p[:,1], p[:,2], lw=1.5, color=c, alpha=0.8,
                label=f'Demo {i+1}')
        ax.scatter(*p[0],  s=30, color=c, marker='o', zorder=5)
        ax.scatter(*p[-1], s=40, color=c, marker='*', zorder=5)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('Stage 2 (PyBullet) — Book-Placing Demonstrations', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    _set_axes_equal(ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_demo_timeseries(demos, save_path="s2_demos_ts.png") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    labels  = ['x', 'y', 'z']
    qlabels = ['w', 'qx', 'qy', 'qz']
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(demos)))
    for i, (d, c) in enumerate(zip(demos, colours)):
        t = d['times'] - d['times'][0]
        for j in range(3):
            axes[0, j].plot(t, d['pos'][:, j], color=c, lw=1.3, alpha=0.8)
            axes[0, j].set_title(f'EE position — {labels[j]}', fontsize=10)
            axes[0, j].set_xlabel('time (s)'); axes[0, j].set_ylabel('m')
            axes[0, j].grid(True, alpha=0.3)
        for j in range(3):
            axes[1, j].plot(t, d['quat'][:, j], color=c, lw=1.3, alpha=0.8,
                            label=f'Demo {i+1}' if j == 0 else None)
            axes[1, j].set_title(f'Quaternion — {qlabels[j]}', fontsize=10)
            axes[1, j].set_xlabel('time (s)'); axes[1, j].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7, loc='best')
    plt.suptitle('Stage 2 (PyBullet) — Position & Orientation Time-Series', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_single_demo_arm(demos, scene,
                         save_path="s2_arm_snapshots.png") -> None:
    d     = demos[0]
    T_    = len(d['times'])
    idxs  = [0, T_//5, T_//2, T_-1]
    lbls  = ['Start', 'Lift', 'Approach', 'Insert']
    fig   = plt.figure(figsize=(18, 5))
    for col, (idx, lbl) in enumerate(zip(idxs, lbls)):
        ax = fig.add_subplot(1, 4, col+1, projection='3d')
        q  = d['q_joints'][idx]
        _, T_list = fk(q)
        draw_arm(ax, T_list)
        draw_frame(ax, T_list[-1], scale=0.06)
        draw_box(ax, make_pose(np.array([0.4, 0., -0.02])),
                 np.array([0.9, 0.8, 0.04]), color='#C8AD7F', alpha=0.2)
        draw_box(ax, scene.T_rack, np.array([0.22, 0.32, 0.32]),
                 color='#8B6914', alpha=0.18)
        draw_frame(ax, scene.T_slot_world, scale=0.05, label='slot')
        ax.plot(d['pos'][:,0], d['pos'][:,1], d['pos'][:,2],
                '--', color='gray', lw=0.8, alpha=0.4)
        ax.scatter(*d['pos'][idx], s=40, color='#D85A30', zorder=5)
        ax.set_title(f'{lbl}\nt={d["times"][idx]:.2f}s', fontsize=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        _set_axes_equal(ax)
    plt.suptitle('Stage 2 (PyBullet) — Arm Snapshots (Demo 1)', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    OUT = os.environ.get("OUT_DIR", ".")
    os.makedirs(OUT, exist_ok=True)

    demos, scene = generate_demonstrations(N_DEMOS=7, seed=42)
    save_demos(demos, os.path.join(OUT, 'demos.npz'))

    passed, total = run_stage2_tests(demos, scene)

    plot_demo_overview(demos, scene,
                       save_path=os.path.join(OUT, "s2_demos_3d.png"))
    plot_demo_timeseries(demos,
                         save_path=os.path.join(OUT, "s2_demos_ts.png"))
    plot_single_demo_arm(demos, scene,
                         save_path=os.path.join(OUT, "s2_arm_snapshots.png"))

    # PyBullet snapshot grid
    visualise_pb_demos(demos, scene,
                       save_path=os.path.join(OUT, "s2_pb_snapshots.png"))

    if passed == total:
        print("\n✓ Stage 2 (PyBullet) COMPLETE")
    else:
        print(f"\n✗ Stage 2: {total-passed} tests failed")
