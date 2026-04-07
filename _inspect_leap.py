import xml.etree.ElementTree as ET
tree = ET.parse('models/leap_rh/leap.urdf')
root = tree.getroot()

print('=== JOINT LIMITS ===')
for j in root.findall('joint'):
    if j.get('type') == 'revolute':
        lim = j.find('limit')
        print(f'  {j.get("name")}: [{lim.get("lower")}, {lim.get("upper")}]')

print()
print('=== DS LINK COLLISIONS ===')
for link in root.findall('link'):
    name = link.get('name')
    if '_ds' in name or name == 'leap_rh_palm':
        for col in link.findall('collision'):
            geom = col.find('geometry')
            origin = col.find('origin')
            for child in geom:
                orig_str = ''
                if origin is not None:
                    orig_str = f' origin xyz={origin.get("xyz","")} rpy={origin.get("rpy","")}'
                print(f'  {name}: {child.tag} {child.attrib}{orig_str}')

print()
print('=== VISUAL MESHES ===')
for link in root.findall('link'):
    name = link.get('name')
    for vis in link.findall('visual'):
        geom = vis.find('geometry')
        mesh_el = geom.find('mesh') if geom is not None else None
        origin = vis.find('origin')
        if mesh_el is not None:
            orig_str = ''
            if origin is not None:
                orig_str = f' origin xyz={origin.get("xyz","")} rpy={origin.get("rpy","")}'
            print(f'  {name}: {mesh_el.get("filename")}{orig_str}')
