"""
GRIPPER CONFIGURATION

Set GRIPPER_START_BODY_NAME and GRIPPER_JOINT_NAMES to match your robot.
Joint and body names are printed on first run.

Open/close values: the env uses the robot's joint limits (soft_joint_pos_limits):
min = open, max = close.
"""

# The only body setting required:
# this body marks where the gripper starts in body_names order.
# The env assumes this body and all following bodies belong to the gripper.
GRIPPER_START_BODY_NAME = "hand_component_1"

GRIPPER_JOINT_NAMES = [
    "Revolute_5",
    "Revolute_6",
    "Revolute_7",
    "Revolute_8",
]

GRIPPER_CLOSE_DISTANCE = 0  # [m]
