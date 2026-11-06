# markov devision progress: here used to return how things calculated:
# e.g. reward function calculation creteria but not reward function itself.

"""Reward calculation criteria for the P1 anti-interference task.

Implement functions from left to right: how rewards are calculated
(source: specification/mdp.md, Implementation mode).
"""

import torch


def tire_lift_penalty(air_time: torch.Tensor, grace_s: float) -> torch.Tensor:
  """I want tire to stay on ground (chat, 2026-11-05; replaces max(0, tair-1)^2).

  Agent: number of tires that have been off the ground for longer than grace_s.
  With weight -1 in env_cfgs.py this costs 1 per tire per second off the ground,
  at all times (not only after a push). The short grace ignores contact bounces.
  """
  # air_time: time each tire has been off the ground [s], shape (num_envs, 4)
  # (mjlab ContactData.current_air_time, 0 while touching). Returns (num_envs,).
  return (air_time > grace_s).sum(dim=-1).float()


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


def arm_ground_contact(found: torch.Tensor) -> torch.Tensor:
  """lets make arm can touch ground but with less penalty; my intension is make at
  less contact as possible (chat, 2026-11-05)

  Agent: 1 while any arm geom touches the ground, else 0. With weight -5 in
  env_cfgs.py this costs 5 per second of arm contact, from the first touch.
  """
  # found: contact counts of the arm geoms with the ground, shape
  # (num_envs, num_geoms); 0 = no contact. Returns (num_envs,) float.
  return torch.any(found > 0, dim=-1).float()


def update_contact_time(
  contact_time: torch.Tensor, touching: torch.Tensor, dt: float
) -> torch.Tensor:
  """Agent: continuous arm contact time [s]; adds dt while touching, back to 0
  as soon as the arm leaves the ground."""
  # contact_time: (num_envs,) [s]; touching: (num_envs,) bool; dt: step [s]
  return torch.where(touching, contact_time + dt, torch.zeros_like(contact_time))


def arm_contact_too_long(contact_time: torch.Tensor, tolerance_s: float) -> torch.Tensor:
  """maybe if tolerance added for arm temporary can touch a bit (chat, 2026-11-05)

  Agent: True (= failure, episode ends) once the arm has touched the ground
  continuously for longer than tolerance_s.
  """
  return contact_time > tolerance_s
