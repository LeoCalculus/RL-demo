# markov devision progress: here used to return how things calculated:
# e.g. reward function calculation creteria but not reward function itself.

"""Reward calculation criteria for the P1 anti-interference task.

Implement functions from left to right: how rewards are calculated
(source: specification/mdp.md, Implementation mode).
"""

import torch


def wheel_air_time_penalty(t_air: torch.Tensor) -> torch.Tensor:
  """this function calculates penalties for wheels in air for long time:
  input a session that the wheel in air time, output max(0, tair-1)^2
  """
  # t_air: wheel air time in the session [s], shape (num_envs,)
  return torch.clamp(t_air - 1.0, min=0.0) ** 2


def non_tire_contact_failure(found: torch.Tensor) -> torch.Tensor:
  """this function tells if using limb for support consider as failure
  this function tells if arm contact with ground consider as failure:
  this function tells if fell over consider as failure:
  Input is non tire contact with ground?
  """
  # found: contact counts of the non-tire geoms with the ground, shape
  # (num_envs, num_geoms); 0 = no contact (mjlab ContactData.found).
  # Returns True where any non-tire geom touches the ground, shape (num_envs,).
  return torch.any(found > 0, dim=-1)


def imu_upright_reward(z: torch.Tensor) -> torch.Tensor:
  """this function reads from IMU on the base link to check whether inclined:
  input IMU z reading, output: 1-(z/9.81+1)^2
  """
  # z: IMU gravity reading along its z axis [m/s^2]; -9.81 when upright
  return 1.0 - (z / 9.81 + 1.0) ** 2


def hip_return_reward(
  q: torch.Tensor, q_ref: torch.Tensor, std: float
) -> torch.Tensor:
  """this function rewards if robot can go back to its original joint value place
  after disturbance
  hip return was actually part of go back to original pose, I just wanna in case
  it learned like using kneel to rasie the CoM to do reward hacking

  Agent: formula exp(-mean(((q - q_ref) / std)^2)) is proposed by the agent
  (the note has none); validate with tests/test_mdp.py.
  """
  # q: joint positions [rad], shape (num_envs, num_joints); q_ref: reference
  # positions in the same joint order [rad]; std: allowed error scale [rad].
  # Returns 1 at the reference pose, falling toward 0 as the pose moves away.
  return torch.exp(-torch.mean(((q - q_ref) / std) ** 2, dim=-1))


def com_height_reward(h: torch.Tensor, h_ref: float, std: float) -> torch.Tensor:
  """this function rewards if CoM was mantained by leg:

  Agent: formula exp(-((h - h_ref) / std)^2) is proposed by the agent
  (the note has none); validate with tests/test_mdp.py.
  """
  # h: whole-robot CoM height above the ground [m], shape (num_envs,);
  # h_ref: reference height [m]; std: allowed error scale [m].
  return torch.exp(-(((h - h_ref) / std) ** 2))


def late_recovery_penalty(t: torch.Tensor) -> torch.Tensor:
  """this function penalize if recover too late: (t-5)^2 for t>=5"""
  # t: elapsed recovery time [s]; no penalty before the 5 s deadline
  return torch.clamp(t - 5.0, min=0.0) ** 2
