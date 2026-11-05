from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  p1_anti_interference_cfg
)
from .rl_cfg import p1_ppo_runner_cfg

# tasks, aka profile for the robot RL

register_mjlab_task(
  task_id="P1-anti-interference",
  env_cfg=p1_anti_interference_cfg(),
  play_env_cfg=p1_anti_interference_cfg(play=True),
  rl_cfg=p1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

