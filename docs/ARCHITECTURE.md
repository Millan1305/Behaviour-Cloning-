See README.md "Architecture". Data flow: landmarks (12x4) -> features (12) -> [retarget -> labels (8)] -> BCNet -> joint targets (8 canonical)
-> h1_controller (config/h1_joints.yaml mapping, URDF limits, EMA + rate limit) -> Float64MultiArray (19 H1 joints) -> gz_ros2_control.
