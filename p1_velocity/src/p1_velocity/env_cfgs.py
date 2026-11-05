"""P1 velocity environment configurations."""

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from p1_velocity import mdp as p1_mdp
from p1_velocity.p1.p1_constants import (
  ARM_JOINTS,
  LEG_JOINTS,
  LEGS,
  get_p1_robot_cfg,
)

##
# Inputs: fill in or check these numbers.
##

# Reward weights: put your number in place of None (None = term left out).
# Recommendation beside each. mjlab multiplies every term by the 0.02 s step, so a
# term worth 1 with weight 1 adds 1 per second of episode.
REWARD_WEIGHTS: dict[str, float | None] = {
  "upright": 1,  
  "leg_pose": 1,  
  "com_height": 1,  
  "wheel_air_time": -1,  
  "late_recovery": -1,  
  "failure": -50,  
  "action_rate_l2": -0.1, 
  "dof_pos_limits": -1.0,  
}

# Other numbers your notes leave open, pre-filled with recommendations.
ACTION_SCALE = 0.06  # rad per action unit = 0.25 * 97 / 400 (template had 0.5)
LEG_POSE_STD = 0.1  # rad, width of the leg pose reward
COM_HEIGHT_REF = 0.3338  # m, settled whole-robot CoM height measured in this env
COM_HEIGHT_STD = 0.02  # m, width of the CoM height reward
RECOVERED_TOL = 0.1  # rad, all 12 leg joints this close to standing = recovered
RECOVERY_WINDOW_S = 5.0  # s, recovery window from your diagram
# N, max push size along the robot's forward, sideways (arm side), up axes
PUSH_FORCE_MAX_N = (150.0, 100.0, 150.0)
PUSH_DURATION_S = (0.1, 0.2)  # s, how long one push lasts
PUSH_WAIT_S = (1.0, 2.0)  # s, wait before a push (after a push: window + this)

##
# P1 names.
##

GRAVITY = 9.81  # m/s^2, as in your upright formula
# The four tire collision geoms; every other collision geom is "non-tire".
TIRE_GEOMS = tuple(f"{leg}_wheel_collision" for leg in LEGS)
# Wheel-centre sites added in p1_constants.get_spec().
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)


##
# Env-side terms: measure from the simulation and call the calculations in mdp.py.
##


def _uniform(bounds: tuple[float, float], n: int, device) -> torch.Tensor:
  low, high = bounds
  return torch.rand(n, device=device) * (high - low) + low


def _body_masses(env, asset) -> torch.Tensor:
  masses = torch.as_tensor(
    env.sim.mj_model.body_mass, dtype=torch.float32, device=env.device
  )
  return masses[asset.indexing.body_ids]


def _whole_robot_com(asset, masses: torch.Tensor) -> torch.Tensor:
  """Whole-robot CoM position in the world frame [m], shape (num_envs, 3)."""
  com = asset.data.body_com_pos_w * masses[None, :, None]
  return com.sum(dim=1) / masses.sum()


def push_force_limit(direction: torch.Tensor, force_max) -> torch.Tensor:
  """Largest allowed push [N] along each unit direction (n, 3) in the robot frame.

  force_max = max along (forward, sideways, up); diagonal directions get the limit
  in between (an ellipsoid through those three values).
  """
  force_max = torch.as_tensor(force_max, dtype=direction.dtype, device=direction.device)
  return 1.0 / torch.sqrt(((direction / force_max) ** 2).sum(dim=-1))


def _push_term(env):
  """The push event, or None when pushes are off (e.g. play mode)."""
  try:
    term = env.event_manager.get_term_cfg("push_robot").func
  except ValueError:
    return None
  return term if isinstance(term, ComImpulsePush) else None


def hold_default_joint_targets(env, env_ids, asset_cfg: SceneEntityCfg) -> None:
  """Hold the selected joints at their configured (initial) pose.

  mjlab zeroes every actuator target at reset and only action terms write them
  afterwards, so joints outside the policy action need this reset event to keep
  the pose from p1_constants.py instead of moving to 0 rad.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=env.device)
  target = asset.data.default_joint_pos[env_ids][:, joint_ids]
  asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


class ComImpulsePush(ManagerTermBase):
  """push is like impluse, force well I think 1N-10N? not quite sure,  yeah kinda
  from center of mass since that is obvious? I need a sphere
  just clamp the force in hand side 0-100 and the other without hand 0-150+

  Agent: each push is a force with a random direction on a sphere (in the robot's
  heading frame) and a random size from 0 up to push_force_limit for that
  direction: 100 N sideways (the arm sides), 150 N forward/back and up/down. It is
  applied at the whole-robot CoM (force on base_link plus the torque that moves it
  to the CoM) for duration_s. The first push of an episode waits wait_s; the next
  one also waits out the recovery window, so windows never overlap.
  since_push_end and push_ended feed the recovery terms.
  """

  def __init__(self, cfg: EventTermCfg, env):
    super().__init__(env)
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._base = self._asset.find_bodies("base_link")[0][0]
    self._masses = _body_masses(env, self._asset)
    self._wait_s = cfg.params["wait_s"]
    n = env.num_envs
    self._active = torch.zeros(n, dtype=torch.bool, device=env.device)
    self._time_left = torch.zeros(n, device=env.device)
    self._wait = _uniform(self._wait_s, n, env.device)
    # Seconds since the last push ended; -1 = no push yet this episode.
    self.since_push_end = torch.full((n,), -1.0, device=env.device)
    # True on the step a push ended.
    self.push_ended = torch.zeros(n, dtype=torch.bool, device=env.device)

  def _write(self, env_ids: torch.Tensor, force: torch.Tensor) -> None:
    """Apply force (n, 3) at the whole-robot CoM through base_link."""
    com = _whole_robot_com(self._asset, self._masses)[env_ids]
    base_com = self._asset.data.body_com_pos_w[env_ids, self._base]
    torque = torch.cross(com - base_com, force, dim=-1)
    self._asset.write_external_wrench_to_sim(
      force.unsqueeze(1), torque.unsqueeze(1), env_ids=env_ids, body_ids=[self._base]
    )

  def __call__(self, env, env_ids, force_max, duration_s, wait_s, window_s, asset_cfg):
    del env_ids, wait_s, asset_cfg  # Step events act on all envs; wait_s read at init.
    dt = env.step_dt
    self.push_ended[:] = False
    self.since_push_end[self.since_push_end >= 0] += dt

    # End pushes whose duration is over.
    self._time_left[self._active] -= dt
    ended = self._active & (self._time_left <= 0)
    if ended.any():
      ids = ended.nonzero().squeeze(-1)
      self._write(ids, torch.zeros(len(ids), 3, device=self.device))
      self._active[ids] = False
      self.push_ended[ids] = True
      self.since_push_end[ids] = 0.0
      self._wait[ids] = window_s + _uniform(self._wait_s, len(ids), self.device)

    # Start new pushes.
    self._wait[~self._active] -= dt
    start = ~self._active & (self._wait <= 0)
    if start.any():
      ids = start.nonzero().squeeze(-1)
      n = len(ids)
      direction = torch.randn(n, 3, device=self.device)
      direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
      size = torch.rand(n, device=self.device) * push_force_limit(direction, force_max)
      force_h = direction * size.unsqueeze(-1)  # robot heading frame
      # Heading frame -> world: rotate by the robot's yaw.
      yaw = self._asset.data.heading_w[ids]
      cos, sin = torch.cos(yaw), torch.sin(yaw)
      force = torch.stack(
        (
          cos * force_h[:, 0] - sin * force_h[:, 1],
          sin * force_h[:, 0] + cos * force_h[:, 1],
          force_h[:, 2],
        ),
        dim=-1,
      )
      self._write(ids, force)
      self._active[ids] = True
      self._time_left[ids] = _uniform(duration_s, n, self.device)

  def reset(self, env_ids=None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)
    active = env_ids[self._active[env_ids]]
    if len(active) > 0:
      self._write(active, torch.zeros(len(active), 3, device=self.device))
    self._active[env_ids] = False
    self._time_left[env_ids] = 0.0
    self._wait[env_ids] = _uniform(self._wait_s, len(env_ids), self.device)
    self.since_push_end[env_ids] = -1.0
    self.push_ended[env_ids] = False


def upright_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """mdp.imu_upright_reward on the IMU z gravity reading (IMU axes = base_link)."""
  asset = env.scene[asset_cfg.name]
  z = GRAVITY * asset.data.projected_gravity_b[:, 2]
  return p1_mdp.imu_upright_reward(z)


def leg_pose_reward(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """for all 12 would be better at its original given pos

  Agent: mdp.hip_return_reward on the 12 leg joints, reference = standing pose.
  """
  asset = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_ref = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return p1_mdp.hip_return_reward(q, q_ref, std)


class ComHeightReward(ManagerTermBase):
  """mdp.com_height_reward on the whole-robot CoM height above the flat plane."""

  def __init__(self, cfg: RewardTermCfg, env):
    super().__init__(env)
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._masses = _body_masses(env, self._asset)

  def __call__(self, env, h_ref, std, asset_cfg):
    del asset_cfg
    h = _whole_robot_com(self._asset, self._masses)[:, 2]
    return p1_mdp.com_height_reward(h, h_ref, std)


class WheelAirTimePenalty(ManagerTermBase):
  """mdp.wheel_air_time_penalty per push: tair = time in the window after a push
  ends with at least one tire off the ground.

  Agent: charged as it builds up (the step's increase of max(0, tair-1)^2, divided
  by dt), so each push costs max(0, tair-1)^2 in total, even if the episode ends
  before the window does.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    super().__init__(env)
    self._t_air = torch.zeros(env.num_envs, device=env.device)
    self._in_window = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env, window_s, sensor_name):
    push = _push_term(env)
    if push is None:
      return torch.zeros(env.num_envs, device=env.device)
    self._t_air[push.push_ended] = 0.0
    self._in_window |= push.push_ended
    self._in_window &= push.since_push_end <= window_s
    lifted = (env.scene[sensor_name].data.found <= 0).any(dim=-1)
    before = p1_mdp.wheel_air_time_penalty(self._t_air)
    self._t_air += env.step_dt * (self._in_window & lifted)
    after = p1_mdp.wheel_air_time_penalty(self._t_air)
    return (after - before) / env.step_dt

  def reset(self, env_ids=None) -> None:
    self._t_air[env_ids] = 0.0
    self._in_window[env_ids] = False


class LateRecoveryPenalty(ManagerTermBase):
  """went back to pose joint var value with tolerance

  Agent: after a push ends, t counts until all 12 leg joints are within
  `tolerance` of the standing pose (= recovered); mdp.late_recovery_penalty(t)
  applies each step until then.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    super().__init__(env)
    self._waiting = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env, tolerance, asset_cfg):
    push = _push_term(env)
    if push is None:
      return torch.zeros(env.num_envs, device=env.device)
    self._waiting |= push.push_ended
    asset = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
      :, asset_cfg.joint_ids
    ]
    recovered = (error.abs() <= tolerance).all(dim=-1)
    self._waiting &= ~recovered
    penalty = p1_mdp.late_recovery_penalty(push.since_push_end)
    return torch.where(self._waiting, penalty, torch.zeros_like(penalty))

  def reset(self, env_ids=None) -> None:
    self._waiting[env_ids] = False


def p1_anti_interference_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  r"""Import necessary libs

  start functon env function for robot

  here implement declear the robot on a flat plane, always using the flat plane in this case

  starting from initial position of joint angle for robot to stand as angles stated in p1_constants.py
  the height value is confirmed in p1_constants can be used directly.

  simulation use policy iteration at 50Hz while Mujoco use its physics at 500Hz, each episoid max
  should have 5s, and total environment counts can be 4096?\

  Actions, for wheel joints they dont need the velocity

  rewards, deal with custom rewards in mdp.py

  arms and passive joints, we only need to keep arm away from the ground and make sure no invalid contact

  command, only need to stand with original joint variable values, not need original place

  imu on base joint as required, actually all three plates of folding mechanism need 1 IMU due to
  concern on actual RH17 encoder precision

  disturbance: apply to the center of robot, direction can be all direction (a ball) and the magnitude
  also from seed but the force should not too much that impossible for robot to recover

  Termination: invalid contact, also a roll/top over need considered for early stopping
  """
  cfg = make_velocity_env_cfg()

  # Solver buffers sized for this flat-ground task (mjlab Go1-flat values). The
  # template's 500 / 500 / default 1500 needed ~18 GB of GPU memory at 4096 envs;
  # see training_issues.md in the main folder.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 50
  cfg.sim.njmax = 300

  # Physics 500 Hz, policy 50 Hz (0.002 s x 10), 5 s episodes, 4096 envs.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  cfg.episode_length_s = 5.0
  cfg.scene.num_envs = 4096

  cfg.scene.entities = {"robot": get_p1_robot_cfg()}

  # Flat plane only: drop the rough-terrain generator and everything that needs it.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  sensors = []
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      continue  # Height scan is not needed on a flat plane.
    if sensor.name == "foot_height_scan":
      # Critic wheel height: measured from the wheel-centre sites.
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site, entity="robot") for site in FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
    sensors.append(sensor)

  tire_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",  # Name used by the critic's foot observations.
    primary=ContactMatch(mode="geom", pattern=TIRE_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  non_tire_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Every collision geom (arms, body, legs, wheel hubs) except the 4 tires.
      pattern=r".*_collision.*",
      exclude=TIRE_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (*sensors, tire_ground_cfg, non_tire_ground_cfg)

  # Spawn at the confirmed height from p1_constants (no extra drop).
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  # Actions: the policy drives the 12 leg joints only. Wheels get no action and
  # keep a zero velocity target; arms are held at their configured pose.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINTS
  joint_pos_action.scale = ACTION_SCALE
  cfg.events["hold_arm_pose"] = EventTermCfg(
    func=hold_default_joint_targets,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
  )

  # Command: stand only, no velocity command and no return to the start position.
  cfg.commands = {}
  cfg.curriculum.pop("command_vel", None)
  del cfg.observations["actor"].terms["command"]
  del cfg.observations["critic"].terms["command"]

  # Disturbance: impulse at the whole-robot CoM, random direction on a sphere.
  cfg.events["push_robot"] = EventTermCfg(
    func=ComImpulsePush,
    mode="step",
    params={
      "force_max": PUSH_FORCE_MAX_N,
      "duration_s": PUSH_DURATION_S,
      "wait_s": PUSH_WAIT_S,
      "window_s": RECOVERY_WINDOW_S,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = TIRE_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Rewards: drop the velocity-task rewards, then add the terms whose weight is set.
  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    cfg.rewards.pop(name, None)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0

  legs = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  custom_terms = {
    "upright": (upright_reward, {"asset_cfg": SceneEntityCfg("robot")}),
    "leg_pose": (leg_pose_reward, {"std": LEG_POSE_STD, "asset_cfg": legs}),
    "com_height": (
      ComHeightReward,
      {
        "h_ref": COM_HEIGHT_REF,
        "std": COM_HEIGHT_STD,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "wheel_air_time": (
      WheelAirTimePenalty,
      {"window_s": RECOVERY_WINDOW_S, "sensor_name": tire_ground_cfg.name},
    ),
    "late_recovery": (
      LateRecoveryPenalty,
      {"tolerance": RECOVERED_TOL, "asset_cfg": legs},
    ),
    "failure": (envs_mdp.is_terminated, {}),
  }
  for name, weight in REWARD_WEIGHTS.items():
    if weight is None:
      cfg.rewards.pop(name, None)
    elif name in custom_terms:
      func, params = custom_terms[name]
      cfg.rewards[name] = RewardTermCfg(func=func, weight=weight, params=params)
    else:
      cfg.rewards[name].weight = weight

  # Terminations: any non-tire ground contact; the template's fell_over
  # (tilt > 70 deg) stays for roll / tip-over.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": non_tire_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


# Open questions and my notes for this config: see specification/env_cfg.md,
# Implementation mode, under the note.
