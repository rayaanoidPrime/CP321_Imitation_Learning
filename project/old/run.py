from pb_stage2_demos import generate_demonstrations, save_demos
from pb_stage3_emp import preprocess_demo, train_emp, adapt_emp, rollout_se3
from pb_stage2_demos import load_demos
import numpy as np
from pb_stage1_env import PandaSimEnv
from pb_utils import make_pose
from pb_stage1_env import ik
import time

# =========================
# STEP 1: Generate demos
# =========================
print("Generating demonstrations...")
demos, scene = generate_demonstrations(N_DEMOS=3)
save_demos(demos, "demos.npz")

# =========================
# STEP 2: Train EMP model
# =========================
print("Training EMP model...")
data = preprocess_demo(demos[0])
model = train_emp(data)

# =========================
# STEP 3: Adapt to new goal
# =========================
print("Adapting policy...")
new_start = demos[0]['pos'][0]
new_goal  = np.array([0.6, -0.1, 0.2])

adapted_model = adapt_emp(model, new_start, new_goal)

# =========================
# STEP 4: Rollout trajectory
# =========================
print("Rolling out trajectory...")
pos_traj, quat_traj = rollout_se3(
    new_start,
    model['Q_train'][0],
    adapted_model['means'], adapted_model['covs'], adapted_model['priors'],
    adapted_model['Ak_pos'], adapted_model['x_star'],
    adapted_model['mu3d'], adapted_model['cov3d'], adapted_model['priors_o'],
    adapted_model['Ak_ori'], adapted_model['q_star']
)

print("Trajectory generated:", pos_traj.shape)
print("Done ✅")

# =========================
# STEP 5: Visualize Demo vs Adapted
# =========================
from pb_stage1_env import PandaSimEnv, ik
from pb_utils import make_pose
import time

print("Visualizing Demo → Adapted...")

with PandaSimEnv(gui=True) as sim:
    time.sleep(1)

    # =========================
    # 1. PLAY DEMO TRAJECTORY
    # =========================
    print("Playing DEMO trajectory...")

    for q in demos[0]['q_joints']:
        sim.set_joint_angles(q)
        sim.step()
        time.sleep(0.05)   # 👈 normal speed

    time.sleep(2)

    # =========================
    # 2. PLAY ADAPTED TRAJECTORY
    # =========================
    print("Playing ADAPTED trajectory...")

    q_cur = demos[0]['q_joints'][0].copy()

    for pos in pos_traj:
        T = make_pose(pos)
        q_cur, _, _ = ik(T, q0=q_cur)

        sim.set_joint_angles(q_cur)
        sim.step()
        time.sleep(0.05)   # 👈 same speed

    print("Done viewing ✅")
    input("Press Enter to exit...")