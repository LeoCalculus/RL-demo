# constants from p1 previously known from anymal_c_constants.py
# source: https://github.com/mujocolab/anymal_c_velocity/blob/main/src/anymal_c_velocity/anymal_c/anymal_c_constants.py

from pathlib import Path

import mujoco
import numpy as np
import math
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg, GeomCfg

# Pre-compile process

##
# Joint groups
##

LEGS = ("right_front", "left_front", "right_rear", "left_rear")
LEG_JOINTS = tuple(
  f"{leg}_{joint}" for leg in LEGS for joint in ("hip_swing", "thigh_pitch", "knee_motor")
)
ARM_JOINTS = tuple(
  f"{side}{joint}" for side in ("", "left_") for joint in ("arm_deploy", "arm_pitch", "hand_roll")
)
WHEEL_JOINTS = tuple(f"{leg}_wheel_spin" for leg in LEGS)

##
# MJCF configuration
##

# root folder place
_HERE = Path(__file__).resolve().parents[4]

# Use p1_rl, it has less collision surface than p1_reconfigurable
P1_RL_XML: Path = _HERE / "robot_model" / "mujoco" / "p1_rl.xml"
assert P1_RL_XML.exists()

# cad frame we have x right, y up and -z forward, convert to mjlab frame 
# C_mj = C_cad cad_T_mj 
_CAD_TO_BASE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def _rotate_base_frame(spec: mujoco.MjSpec) -> None:
  # find base link first
  base = spec.body("base_link")
  rotation = np.zeros(4)
  # convert to quaternion
  mujoco.mju_mat2Quat(rotation, _CAD_TO_BASE.flatten())
  # go through each child joint
  for child in (*base.geoms, *base.bodies):
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, rotation, child.quat)
    # update both orientation and corresponding position
    child.pos = _CAD_TO_BASE @ child.pos
    child.quat = quat
  # here update ineratia
  ixx, iyy, izz, ixy, ixz, iyz = base.fullinertia
  inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
  inertia = _CAD_TO_BASE @ inertia @ _CAD_TO_BASE.T
  # ixx (0,0); iyy (1,1); izz (2,2); ixy (0,1); ixz (0,2); iyz (1,2)
  base.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
  # update base pose and orientation
  base.ipos = _CAD_TO_BASE @ base.ipos
  base.quat = [1.0, 0.0, 0.0, 0.0]

# get specification on xml stuff, like sensors
def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(P1_RL_XML))
  # mjlab adds its own actuators and terrain, remove them
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  spec.delete(spec.geom("ground"))
  # rotate the model back
  _rotate_base_frame(spec)

  # IMU on the upper plate of the folding mechanism, axes along the base frame.
  plate = spec.geom("base_link_collision_top_plate")
  imu_pos = plate.pos + [0.0, 0.0, min(plate.size)]
  spec.body("base_link").add_site(name="imu", pos=imu_pos, group=5)
  for name, sensor in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),):
    spec.add_sensor(name=name, type=sensor, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(
    name="root_angmom", # need total angular momentum
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="base_link",
  )
  for leg in LEGS:
    # wheel centre, so the site does not move as the wheel spins => otherwise the height oscillates
    centre = spec.geom(f"{leg}_wheel_collision").pos
    # site is a position + orientation
    spec.body(f"{leg}_wheel").add_site(name=f"{leg}_foot", pos=centre, group=5)
  return spec

##
# Actuator config.
##

# most motors using the same type, assign them in terms of groups
# the LEG need to controlled by torque: which use MIT mode -> mjlab IdealPd
# stiffness -> Kp; damping -> Kd
P1_LEG_ACTUATOR = IdealPdActuatorCfg(
  target_names_expr=LEG_JOINTS,
  stiffness=400.0,
  damping=4.0,
  effort_limit=97.0,
)

# The arm manipulation need precise position
P1_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=ARM_JOINTS,
  stiffness=40.0,
  damping=0.3,
  effort_limit=40.0,
)

# The wheel only needs to control the speed 
P1_WHEEL_ACTUATOR = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  damping=0.15,
  effort_limit=4.5,
)

M3508_SPEED_LOOP_GAIN = 1.0
M3508_ROTOR_INERTIA = 0.0065
M3508_CONTINUOUS_TORQUE = 3.0
P1_WHEEL_ACTUATOR_M3508 = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  damping=M3508_SPEED_LOOP_GAIN,
  effort_limit=4.5,
  armature=M3508_ROTOR_INERTIA,
)

# keyframes: important frames need to use as reference
# this was found from simulators by human tuning
# hand only need to make sure not close to ground
STAND_ANGLE_DEG = {
  "right_rear_hip_swing": 18.0712,
  "right_rear_thigh_pitch": 40.0,
  "right_rear_knee_motor": -20.0,
  "right_front_hip_swing": 0.0,
  "right_front_thigh_pitch": -43.33618156,
  "right_front_knee_motor": -100.3796155,
  "left_front_hip_swing": -0.0,
  "left_front_thigh_pitch": 43.33618156,
  "left_front_knee_motor": 100.3796155,
  "left_rear_hip_swing": -18.0712,
  "left_rear_thigh_pitch": -40.0,
  "left_rear_knee_motor": 20.0,
  "arm_deploy": -150.0,
  "left_arm_deploy": 150.0,
}

# convert from degrees to radians
def pose_joint_pos(angles_deg: dict[str, float]) -> dict[str, float]:
  pos = {name: math.radians(deg) for name, deg in angles_deg.items()}
  for leg in LEGS:
    knee = pos[f"{leg}_knee_motor"]
    pos[f"{leg}_auto_knee_linkage"] = knee
    pos[f"{leg}_auto_pushrod_linkage"] = -knee
  return pos

STAND_JOINT_POS = pose_joint_pos(STAND_ANGLE_DEG)

INIT_STATE = EntityCfg.InitialStateCfg(
  # from CAD we have height 0.55912362462219056 and need to be lower, hand-tuned here
  pos=(0.0, 0.0, 0.487424),
  joint_pos=STAND_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(right|left)_(front|rear)_wheel_collision$"
_collision_regex = r".*_collision.*"

# profile for full collision
FULL_COLLISION = GeomCfg(
  geom_names_expr=(_collision_regex,),
  # collision for normal force only, for wheel need normal force + friction
  condim={_foot_regex: 3, _collision_regex: 1},
  # wheel to terrain use the wheel's own friction
  priority={_foot_regex: 1},
  # sliding friction coefficient
  friction={_foot_regex: (0.6,)},
  # contact softness
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
)

##
# Final config.
##

P1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(P1_LEG_ACTUATOR, P1_ARM_ACTUATOR, P1_WHEEL_ACTUATOR,),
  # use soft range for joint xml range, used by rewards and reset
  soft_joint_pos_limit_factor=0.9,
)

# get a p1 robot configuration instance
def get_p1_robot_cfg() -> EntityCfg:
  return EntityCfg(init_state=INIT_STATE, geoms=(FULL_COLLISION,), 
                   spec_fn=get_spec, articulation=P1_ARTICULATION,)

# the viewer to see the robot 
if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity
  robot = Entity(get_p1_robot_cfg())
  viewer.launch(robot.spec.compile())