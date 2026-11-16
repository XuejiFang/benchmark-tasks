"""Generate two nestable breakfast bowls with open, colliding interiors."""

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from build_key_return_scene import box, stage_at


# The first two profile points form the outer bottom. The remaining points
# pass over the rolled lip and down to the raised inner bottom.
PROFILE = (
    (.046, .000), (.051, .003), (.064, .017), (.078, .037),
    (.079, .043), (.076, .046), (.073, .044), (.073, .038),
    (.059, .019), (.048, .008), (.046, .008),
)


def mesh(stage, name, points, faces, color=None):
    prim = UsdGeom.Mesh.Define(stage, name)
    prim.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    prim.CreateFaceVertexCountsAttr([len(face) for face in faces])
    prim.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    prim.CreateSubdivisionSchemeAttr('none')
    if color:
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim


def polar(radius, z, angle):
    return radius * math.cos(angle), radius * math.sin(angle), z


def collision(prim, approximation):
    api = UsdPhysics.CollisionAPI.Apply(prim)
    api.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def bowl(output, name, color):
    stage = stage_at(output / f'{name}.usda', 'Bowl')
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.31)
    count = 48
    points = [polar(radius, z, i*math.tau/count) for radius, z in PROFILE for i in range(count)]
    faces = [
        (k*count+i, k*count+(i+1)%count, (k+1)*count+(i+1)%count, (k+1)*count+i)
        for k in range(len(PROFILE)-1) for i in range(count)
    ]
    faces += [tuple(range(count-1, -1, -1)),
              tuple(range((len(PROFILE)-1)*count, len(PROFILE)*count))]
    glaze = mesh(stage, '/Bowl/Glaze', points, faces, color)
    material = UsdShade.Material.Define(stage, '/Bowl/Ceramic')
    shader = UsdShade.Shader.Define(stage, '/Bowl/Ceramic/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.24)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(glaze.GetPrim()).Bind(material)

    # Use a separate solid base and compound convex segments for the wall.
    # A single convex hull would fill the bowl's concavity and prevent stacking.
    base = UsdGeom.Cylinder.Define(stage, '/Bowl/BaseCollider')
    base.CreateRadiusAttr(.048)
    base.CreateHeightAttr(.008)
    base.AddTranslateOp().Set(Gf.Vec3d(0., 0., .004))
    base.CreatePurposeAttr('guide')
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    for i in range(20):
        angles = [(i-.012)*math.tau/20, (i+1.012)*math.tau/20]
        rim = [polar(r, z, a) for a in angles for r, z in PROFILE[1:10]]
        # Collision uses the convex envelope of one angular segment. The
        # central opening is left free by all segment hulls.
        faces = [tuple(range(8, -1, -1)), tuple(range(9, 18))]
        faces += [(j, (j+1)%9, 9+(j+1)%9, 9+j) for j in range(9)]
        collider = mesh(stage, f'/Bowl/RimCollider{i:02}', rim, faces)
        collider.CreatePurposeAttr('guide')
        collision(collider.GetPrim(), 'convexHull')
    stage.GetRootLayer().Save()


def spoon(output, name, color):
    stage = stage_at(output / f'{name}.usda', 'Spoon')
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.041)
    # A molded handle sits on the table and gives the fingertips room to close
    # across its parallel faces without supporting it by a single thin edge.
    box(stage, '/Spoon/Handle', (0., -.031, .004), (.014, .116, .007), color)
    # Bake the ellipsoid into mesh coordinates so PhysX cannot substitute a
    # uniformly scaled sphere, which would lift the flat handle off the table.
    segments = 24
    rims = [(math.sin(angle), math.cos(angle))
            for angle in (math.pi/8, math.pi/4, 3*math.pi/8, math.pi/2,
                          5*math.pi/8, 3*math.pi/4, 7*math.pi/8)]
    vertices = [
        (.021*r*math.cos(a), .05+.032*r*math.sin(a), .004+.004*z)
        for r, z in rims for a in [i*math.tau/segments for i in range(segments)]
    ]
    faces = [
        (j*segments+i, (j+1)*segments+i, (j+1)*segments+(i+1)%segments, j*segments+(i+1)%segments)
        for j in range(len(rims)-1) for i in range(segments)
    ]
    top, bottom = len(vertices), len(vertices)+1
    vertices.extend([(0., .05, .008), (0., .05, 0.)])
    faces += [(top, i, (i+1)%segments) for i in range(segments)]
    index = (len(rims)-1)*segments
    faces += [(bottom, index+(i+1)%segments, index+i) for i in range(segments)]
    head = mesh(stage, '/Spoon/Head', vertices, faces, (.62, .66, .71))
    collision(head.GetPrim(), 'convexHull')
    stage.GetRootLayer().Save()


def stack_probe(output):
    stage = stage_at(output / 'nested_bowls.usda', 'Stack')
    for name, height in [('cream_bowl', .002), ('teal_bowl', .018)]:
        instance = UsdGeom.Xform.Define(stage, f'/Stack/{name}')
        instance.GetPrim().GetReferences().AddReference(f'./{name}.usda')
        instance.AddTranslateOp().Set(Gf.Vec3d(.0, .0, height))
    stage.GetRootLayer().Save()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('output', type=Path)
    output = args.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    bowl(output, 'teal_bowl', (.06, .30, .34))
    bowl(output, 'cream_bowl', (.71, .56, .29))
    spoon(output, 'dark_spoon', (.21, .24, .28))
    spoon(output, 'light_spoon', (.66, .61, .48))
    stack_probe(output)
