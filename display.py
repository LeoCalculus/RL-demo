# display used to help find angle by tuning by hand and check what will happen to robot.
"""Watch the trained P1 policy in MuJoCo while it is pushed left / right.

Red arrow = the push, acting at the whole-robot CoM (arrow length 1 m per 300 N).
Run: cd p1_velocity && uv run python ../display.py
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path

# Use this repo's p1_velocity, not another copy installed in the Python env.
sys.path.insert(0, str(Path(__file__).parent / "p1_velocity" / "src"))

import mujoco
import mujoco.viewer
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from p1_velocity.env_cfgs import _body_masses, _whole_robot_com, p1_anti_interference_cfg
from p1_velocity.rl_cfg import p1_ppo_runner_cfg

# Final checkpoint of the 2026-11-05 vast.ai run (copied from logs/.../vast_5090).
CKPT = Path(__file__).parent / "p1_velocity" / "result" / "model_2999.pt"
FORCE_N = (100.0, 200.0)  # push size range, centred at 150 N (training max was 150 N)
PUSH_S = 0.15  # how long one push lasts
EVERY_S = 3.0  # one push every 3 s, randomly to the left or right

device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = p1_anti_interference_cfg(play=True)
cfg.scene.num_envs = 1
cfg.events.pop("push_robot")  # the pushes come from this script instead
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
runner = VelocityOnPolicyRunner(env, asdict(p1_ppo_runner_cfg()), device=device)
runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

sim, robot = env.unwrapped.sim, env.unwrapped.scene["robot"]
base = robot.find_bodies("base_link")[0][0]
masses = _body_masses(env.unwrapped, robot)
dt = env.unwrapped.step_dt
every, push_steps = round(EVERY_S / dt), round(PUSH_S / dt)
m, d = sim.mj_model, mujoco.MjData(sim.mj_model)
obs, force, step = env.get_observations(), torch.zeros(1, 3, device=device), 0

with mujoco.viewer.launch_passive(m, d) as viewer:
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  viewer.cam.trackbodyid = int(robot.indexing.body_ids[base])
  viewer.cam.distance = 2.0
  while viewer.is_running():
    tic = time.time()
    if step > 0 and step % every == 0:  # new push: left (+) or right (-) of the robot
      sign, size = np.random.choice([1, -1]), np.random.uniform(*FORCE_N)
      yaw = robot.data.heading_w[0]
      force[0] = float(sign * size) * torch.stack((-torch.sin(yaw), torch.cos(yaw), yaw * 0))
      print(f"t = {step * dt:5.1f} s  push {size:3.0f} N to the {'left' if sign > 0 else 'right'}")
    f = force if 0 < step and step % every < push_steps else force * 0

    com = _whole_robot_com(robot, masses)  # force at base_link + torque = force at CoM
    torque = torch.cross(com - robot.data.body_com_pos_w[:, base], f, dim=-1)
    robot.write_external_wrench_to_sim(f[:, None], torque[:, None], body_ids=[base])
    with torch.no_grad():
      obs, _, done, _ = env.step(policy(obs))
    if done[0]:
      print("  -> failed, episode reset")

    d.qpos[:] = sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    viewer.user_scn.ngeom = 0
    if f.any():  # red arrow pointing into the CoM along the push
      end = com[0].cpu().numpy()
      g = viewer.user_scn.geoms[0]
      mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                          np.eye(3).flatten(), np.array([1, 0, 0, 1], np.float32))
      mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02,
                           end - f[0].cpu().numpy() / 300.0, end)
      viewer.user_scn.ngeom = 1
    viewer.sync()
    step += 1
    time.sleep(max(0.0, dt - (time.time() - tic)))
