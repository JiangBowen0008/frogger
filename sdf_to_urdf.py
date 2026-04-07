"""Convert Allegro SDF to URDF format.

SDF link poses are absolute (model frame).  URDF joint origins are
parent-relative.  SDF joint axes with `expressed_in="__model__"` must
be transformed to the child link frame for the URDF.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R


def _pose_to_T(pose_str):
    """Parse SDF pose 'x y z roll pitch yaw' → 4×4 matrix."""
    vals = [float(v) for v in pose_str.strip().split()]
    x, y, z, roll, pitch, yaw = vals
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def _T_to_origin(T):
    """4×4 matrix → URDF <origin xyz="..." rpy="..." />."""
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    return (
        f'<origin xyz="{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}" '
        f'rpy="{rpy[0]:.8g} {rpy[1]:.8g} {rpy[2]:.8g}" />'
    )


def convert(sdf_path, urdf_path, hand="lh"):
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # Strip namespace prefixes for easier parsing
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    model = root.find("model")
    
    # Collect link poses (model frame)  link name -> 4x4
    link_poses = {}
    link_elems = {}
    for link in model.findall("link"):
        name = link.get("name")
        if "FROGGER" in name:
            continue
        link_elems[name] = link
        pose_el = link.find("pose")
        if pose_el is not None and pose_el.text:
            link_poses[name] = _pose_to_T(pose_el.text)
        else:
            link_poses[name] = np.eye(4)

    # Collect joints
    joint_list = []
    for joint in model.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        if jtype == "fixed" or "dummy" in name:
            continue
        parent = joint.find("parent").text.strip()
        child = joint.find("child").text.strip()
        axis_el = joint.find("axis/xyz")
        if axis_el is None:
            continue
        axis_model = np.array([float(v) for v in axis_el.text.strip().split()])
        lim = joint.find("axis/limit")
        lo = float(lim.find("lower").text)
        hi = float(lim.find("upper").text)
        eff_el = lim.find("effort")
        eff = float(eff_el.text) if eff_el is not None else 0.7
        joint_list.append({
            "name": name, "type": jtype,
            "parent": parent, "child": child,
            "axis_model": axis_model,
            "lower": lo, "upper": hi, "effort": eff,
        })

    # Build URDF
    lines = [
        '<?xml version="1.0" ?>',
        f'<robot name="algr_{hand}">',
    ]

    # Emit all links (minimal — just name; visuals/collisions omitted for FK)
    for name, link in link_elems.items():
        lines.append(f'  <link name="{name}">')
        
        # Inertial
        inertial = link.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            mass = float(mass_el.text) if mass_el is not None else 0.01
            inertia_el = inertial.find("inertia")
            ixx = float(inertia_el.find("ixx").text) if inertia_el is not None else 1e-6
            ixy = float(inertia_el.find("ixy").text) if inertia_el is not None else 0
            ixz = float(inertia_el.find("ixz").text) if inertia_el is not None else 0
            iyy = float(inertia_el.find("iyy").text) if inertia_el is not None else 1e-6
            iyz = float(inertia_el.find("iyz").text) if inertia_el is not None else 0
            izz = float(inertia_el.find("izz").text) if inertia_el is not None else 1e-6
            inertial_pose = inertial.find("pose")
            if inertial_pose is not None:
                vals = [float(v) for v in inertial_pose.text.strip().split()]
                lines.append(f'    <inertial>')
                lines.append(f'      <origin xyz="{vals[0]:.8g} {vals[1]:.8g} {vals[2]:.8g}" rpy="{vals[3]:.8g} {vals[4]:.8g} {vals[5]:.8g}" />')
            else:
                lines.append(f'    <inertial>')
                lines.append(f'      <origin xyz="0 0 0" rpy="0 0 0" />')
            lines.append(f'      <mass value="{mass:.8g}" />')
            lines.append(f'      <inertia ixx="{ixx:.8g}" ixy="{ixy:.8g}" ixz="{ixz:.8g}" iyy="{iyy:.8g}" iyz="{iyz:.8g}" izz="{izz:.8g}" />')
            lines.append(f'    </inertial>')

        # Visual(s)
        for vis in link.findall("visual"):
            vis_name = vis.get("name", "visual")
            vis_pose = vis.find("pose")
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is not None:
                fn = mesh_el.find("filename")
                if fn is None:
                    fn = mesh_el.find("uri")
                if fn is not None:
                    lines.append(f'    <visual>')
                    if vis_pose is not None:
                        vals = [float(v) for v in vis_pose.text.strip().split()]
                        lines.append(f'      <origin xyz="{vals[0]:.8g} {vals[1]:.8g} {vals[2]:.8g}" rpy="{vals[3]:.8g} {vals[4]:.8g} {vals[5]:.8g}" />')
                    lines.append(f'      <geometry>')
                    lines.append(f'        <mesh filename="{fn.text.strip()}" />')
                    lines.append(f'      </geometry>')
                    lines.append(f'    </visual>')

        lines.append(f'  </link>')

    # Emit joints with correct relative transforms
    for j in joint_list:
        parent = j["parent"]
        child = j["child"]
        T_parent = link_poses[parent]  # parent pose in model frame
        T_child = link_poses[child]    # child pose in model frame

        # Joint origin = parent^-1 * child  (relative transform)
        T_rel = np.linalg.inv(T_parent) @ T_child

        # Joint axis: transform from model frame to child frame
        R_child = T_child[:3, :3]
        axis_local = R_child.T @ j["axis_model"]
        axis_local = axis_local / np.linalg.norm(axis_local)  # normalize

        lines.append(f'  <joint name="{j["name"]}" type="{j["type"]}">')
        lines.append(f'    <parent link="{parent}" />')
        lines.append(f'    <child link="{child}" />')
        lines.append(f'    {_T_to_origin(T_rel)}')
        lines.append(f'    <axis xyz="{axis_local[0]:.8g} {axis_local[1]:.8g} {axis_local[2]:.8g}" />')
        lines.append(f'    <limit lower="{j["lower"]:.8g}" upper="{j["upper"]:.8g}" effort="{j["effort"]:.8g}" velocity="7.0" />')
        lines.append(f'  </joint>')

    lines.append('</robot>')

    with open(urdf_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote {urdf_path} ({len(lines)} lines)")


if __name__ == "__main__":
    convert(
        "models/allegro/allegro_lh_clean.sdf",
        "models/allegro/allegro_lh.urdf",
        hand="lh",
    )
    convert(
        "models/allegro/allegro_rh_clean.sdf",
        "models/allegro/allegro_rh.urdf",
        hand="rh",
    )
