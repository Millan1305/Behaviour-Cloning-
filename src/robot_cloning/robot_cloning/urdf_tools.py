"""Convert the stock Unitree H1 URDF into a Gazebo Harmonic + gz_ros2_control robot description.

Steps (all done on the XML, original file is never modified):
  1. strip ROS1-era <gazebo>/<transmission> blocks,
  2. rewrite package://h1_description/ mesh URIs to absolute file:// URIs,
  3. add a `world` link and a FIXED joint world->root link (pelvis is held in the air; see README
     'Known limitations': a free-standing biped needs a balance controller, which is out of scope),
  4. add <ros2_control> (position command interface) for all movable joints,
  5. add the gz_ros2_control plugin and write generated controller YAML + joint info YAML.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

MOVABLE = ("revolute", "continuous", "prismatic")


def _inspect(root_el):
    joints, children, links = [], set(), set()
    for l in root_el.findall("link"):
        links.add(l.get("name"))
    for j in root_el.findall("joint"):
        children.add(j.find("child").get("link"))
        if j.get("type") in MOVABLE:
            lim = j.find("limit")
            lo = float(lim.get("lower", -3.14)) if lim is not None else -3.14
            hi = float(lim.get("upper", 3.14)) if lim is not None else 3.14
            if j.get("type") == "continuous":
                lo, hi = -3.14, 3.14
            joints.append((j.get("name"), lo, hi))
    roots = [l for l in links if l not in children]
    return joints, roots


def build_gz_urdf(src_urdf, out_dir, mesh_root, pelvis_height=1.05, controller_update_rate=200, base_link=None):
    """Returns dict(urdf=str(path), controllers=str(path), joint_info=str(path), joint_order=[..])."""
    src_urdf, out_dir, mesh_root = Path(src_urdf), Path(out_dir), Path(mesh_root)
    if not src_urdf.exists():
        raise FileNotFoundError(f"H1 URDF not found: {src_urdf}. Run ./scripts/fetch_h1.sh (or ./setup.sh).")
    tree = ET.parse(src_urdf)
    robot = tree.getroot()
    for tag in ("gazebo", "transmission", "ros2_control"):
        for el in robot.findall(tag):
            robot.remove(el)
    # mesh URIs
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            rest = fn.split("/", 3)[3] if fn.count("/") >= 3 else fn
            mesh.set("filename", f"file://{mesh_root}/{rest}")
    joints, roots = _inspect(robot)
    if not joints:
        raise ValueError("No movable joints found in the H1 URDF - is it the right file?")
    base = base_link or (roots[0] if roots else None)
    if base is None or (base_link and base_link not in [l.get("name") for l in robot.findall("link")]):
        raise ValueError(f"Cannot determine base link (root candidates: {roots})")
    if "world" not in [l.get("name") for l in robot.findall("link")]:
        ET.SubElement(robot, "link", name="world")
    fj = ET.SubElement(robot, "joint", name="world_fixed_joint", type="fixed")
    ET.SubElement(fj, "parent", link="world")
    ET.SubElement(fj, "child", link=base)
    ET.SubElement(fj, "origin", xyz=f"0 0 {pelvis_height}", rpy="0 0 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    controllers = out_dir / "h1_controllers.yaml"
    names = [j[0] for j in joints]

    rc = ET.SubElement(robot, "ros2_control", name="H1GazeboSystem", type="system")
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "gz_ros2_control/GazeboSimSystem"
    for n, lo, hi in joints:
        je = ET.SubElement(rc, "joint", name=n)
        ET.SubElement(je, "command_interface", name="position")
        ET.SubElement(je, "state_interface", name="position")
        ET.SubElement(je, "state_interface", name="velocity")
    gz = ET.SubElement(robot, "gazebo")
    plug = ET.SubElement(gz, "plugin", filename="gz_ros2_control-system",
                         name="gz_ros2_control::GazeboSimROS2ControlPlugin")
    ET.SubElement(plug, "parameters").text = str(controllers.resolve())

    urdf_out = out_dir / "h1_gz.urdf"
    ET.indent(tree, space="  ")
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)

    ctrl = {
        "controller_manager": {"ros__parameters": {
            "update_rate": controller_update_rate,
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            "forward_position_controller": {"type": "position_controllers/JointGroupPositionController"},
        }},
        "forward_position_controller": {"ros__parameters": {"joints": names}},
    }
    controllers.write_text(yaml.safe_dump(ctrl, sort_keys=False))
    info = out_dir / "h1_joint_info.yaml"
    info.write_text(yaml.safe_dump({
        "base_link": base, "joint_order": names,
        "limits": {n: [lo, hi] for n, lo, hi in joints}}, sort_keys=False))
    return dict(urdf=str(urdf_out), controllers=str(controllers), joint_info=str(info), joint_order=names, base_link=base)
