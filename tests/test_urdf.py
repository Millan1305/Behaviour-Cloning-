import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "robot_cloning"))
from robot_cloning.urdf_tools import build_gz_urdf  # noqa: E402

MINI = """<?xml version="1.0"?>
<robot name="mini">
  <link name="pelvis"><visual><geometry><mesh filename="package://h1_description/meshes/pelvis.dae"/></geometry></visual></link>
  <link name="a"/><link name="b"/>
  <joint name="j1" type="revolute"><parent link="pelvis"/><child link="a"/><axis xyz="0 1 0"/><limit lower="-1" upper="2" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 1 0"/><limit lower="-0.5" upper="0.5" effort="1" velocity="1"/></joint>
  <transmission name="t"/>
  <gazebo><plugin filename="libgazebo_ros_control.so" name="x"/></gazebo>
</robot>"""


def test_conversion():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "h1.urdf"
        src.write_text(MINI)
        r = build_gz_urdf(src, Path(d) / "out", "/meshes/h1_description", 1.1)
        t = ET.parse(r["urdf"]).getroot()
        assert r["base_link"] == "pelvis" and r["joint_order"] == ["j1", "j2"]
        assert t.find("transmission") is None
        plugins = [p.get("name") for g in t.findall("gazebo") for p in g.findall("plugin")]
        assert plugins == ["gz_ros2_control::GazeboSimROS2ControlPlugin"]
        assert t.find("ros2_control") is not None and len(t.find("ros2_control").findall("joint")) == 2
        assert t.find(".//mesh").get("filename") == "file:///meshes/h1_description/meshes/pelvis.dae"
        fj = [j for j in t.findall("joint") if j.get("name") == "world_fixed_joint"][0]
        assert fj.find("child").get("link") == "pelvis" and fj.find("origin").get("xyz") == "0 0 1.1"
        assert "forward_position_controller" in Path(r["controllers"]).read_text()
