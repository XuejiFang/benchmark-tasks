"""Build physical tea leaves and open vessels for the T04 settling probe."""

import argparse
import math
from pathlib import Path
import random

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from build_breakfast_assets import mesh
from build_key_return_scene import box, stage_at


def contact(prim, offset):
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema('PhysxCollisionAPI')
    prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(offset)
    prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)


def cylinder(stage, path, center, radius, depth, color):
    part = UsdGeom.Cylinder.Define(stage, path)
    part.CreateRadiusAttr(radius)
    part.CreateHeightAttr(depth)
    part.AddTranslateOp().Set(Gf.Vec3d(*center))
    part.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    contact(part.GetPrim(), .00065)


def vessel(stage, name, center, inner, thickness, height, color):
    """Assemble a hollow vessel out of convex wall sections and a base."""
    prefix = f'{stage.GetDefaultPrim().GetPath()}/{name}'
    x, y = center
    cylinder(stage, prefix+'/Bottom', (x, y, .0025), inner+thickness, .005, color)
    count = 20
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for i in range(count):
        angles = ((i-.006)*math.tau/count, (i+1.006)*math.tau/count)
        points = [(x+r*math.cos(a), y+r*math.sin(a), z)
                  for z in (.004, height)
                  for r,a in ((inner,angles[0]), (inner+thickness,angles[0]),
                              (inner+thickness,angles[1]), (inner,angles[1]))]
        part = mesh(stage, f'{prefix}/Wall{i:02}', points, faces, color)
        contact(part.GetPrim(), .00065)
        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('convexHull')


def leaf(stage, i, position, rng):
    # Approximate rolled loose-leaf tea as individual convex 7-9 mm pieces.
    part = UsdGeom.Mesh.Define(stage, f'{stage.GetDefaultPrim().GetPath()}/Leaf{i:02}')
    size = rng.uniform(.0035, .0045)
    points = [(-size, 0., 0.), (size, 0., .001),
              (-.001, -.002, .0002), (.002, .002, .0002),
              (0., .0, -.0012), (.0005, 0., .0014)]
    triangles = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
                 (0, 5, 2), (2, 5, 1), (1, 5, 3), (3, 5, 0)]
    part.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    part.CreateFaceVertexCountsAttr([3]*len(triangles))
    part.CreateFaceVertexIndicesAttr([j for face in triangles for j in face])
    part.CreateSubdivisionSchemeAttr('none')
    part.AddTranslateOp().Set(Gf.Vec3d(*position))
    part.AddRotateXYZOp().Set(Gf.Vec3f(0., 0., rng.uniform(-180.,180.)))
    shade = rng.uniform(.65, 1.15)
    part.CreateDisplayColorAttr([Gf.Vec3f(.15*shade, .16*shade, .045*shade)])
    prim = part.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.000042)
    contact(prim, .00048)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    prim.AddAppliedSchema('PhysxRigidBodyAPI')
    prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute('physxRigidBody:maxDepenetrationVelocity', Sdf.ValueTypeNames.Float).Set(.18)
    prim.CreateAttribute('physxRigidBody:linearDamping', Sdf.ValueTypeNames.Float).Set(.12)
    prim.CreateAttribute('physxRigidBody:angularDamping', Sdf.ValueTypeNames.Float).Set(.24)


def build(path, count=48, accessories=False):
    stage = stage_at(path, 'TeaStation')
    vessel(stage, 'GreenTeaTin', (0., 0.), .038, .0025, .055, (.20, .39, .27))
    if accessories:
        vessel(stage,'DrinkingCup',(.12,.09),.035,.003,.048,(.48,.32,.22))
    else:
        vessel(stage, 'FilterBasket', (.145, -.017), .029, .0025, .033, (.55, .56, .59))
    rng = random.Random(4)
    for i in range(count):
        column, row, level = i%4, i//4%4, i//16
        spacing = .008 if count == 48 else .0045
        leaf(stage, i, ((column-1.5)*.012, (row-1.5)*.010, .015+level*spacing), rng)
    stage.GetRootLayer().Save()


def build_accessories(folder):
    lid = stage_at(folder/'cap.usda','Cap')
    cap = lid.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(cap)
    UsdPhysics.MassAPI.Apply(cap).CreateMassAttr(.027)
    cylinder(lid,'/Cap/Plate',(0.,0.,.003),.0415,.006,(.20,.39,.27))
    cylinder(lid,'/Cap/InnerRim',(0.,0.,-.0015),.035,.003,(.20,.39,.27))
    cylinder(lid,'/Cap/Knob',(0.,0.,.016),.009,.020,(.19,.23,.22))
    lid.GetRootLayer().Save()
    filter_stage = stage_at(folder/'filter.usda','Infuser')
    basket = filter_stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(basket)
    UsdPhysics.MassAPI.Apply(basket).CreateMassAttr(.037)
    vessel(filter_stage,'Screen',(0.,0.),.029,.0025,.033,(.55,.56,.59))
    for side,x in [('Near',-.032),('Far',.032)]:
        box(filter_stage,'/Infuser/Ear'+side,(x,0.,.033),(.019,.019,.003),(.57,.60,.61))
    box(filter_stage,'/Infuser/GraspTab',(-.039,0.,.043),(.011,.014,.019),(.28,.31,.32))
    for part in basket.GetChildren():
        if part.GetName() != 'Screen':
            contact(part,.00065)
    filter_stage.GetRootLayer().Save()


def build_scoop(path, kinematic=True):
    """A spoon cavity with an entry bevel and open pouring lip."""
    stage = stage_at(path, 'Scoop')
    spoon = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(spoon).CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(spoon).CreateMassAttr(.034)
    silver = (.59, .64, .68)
    box(stage, '/Scoop/Floor', (-.002, 0., .001), (.025, .026, .002), silver)
    box(stage, '/Scoop/Heel', (-.014, 0., .0055), (.002, .026, .009), silver)
    for side, y in [('Left', .0125), ('Right', -.0125)]:
        box(stage, '/Scoop/'+side, (-.002, y, .0045), (.025, .002, .007), silver)
    box(stage, '/Scoop/Handle', (-.013, 0., .040), (.008, .009, .068), (.26,.33,.37))
    profile = [( .0105, .0000), (.0160, .0000), (.0105, .0020)]
    vertices = [(x, y, z) for y in (-.013, .013) for x,z in profile]
    faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
    bevel = mesh(stage, '/Scoop/Bevel', vertices, faces, silver).GetPrim()
    contact(bevel, .00048)
    UsdPhysics.MeshCollisionAPI.Apply(bevel).CreateApproximationAttr('convexHull')
    for part in spoon.GetChildren():
        contact(part, .00048)
    stage.GetRootLayer().Save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--scoop', type=Path)
    parser.add_argument('--dynamic-scoop', action='store_true')
    parser.add_argument('--accessories', action='store_true')
    parser.add_argument('--count', type=int, choices=(48,160), default=48)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.output,args.count,accessories=args.accessories)
    if args.scoop:
        args.scoop.parent.mkdir(parents=True, exist_ok=True)
        build_scoop(args.scoop, kinematic=not args.dynamic_scoop)
    if args.accessories:
        build_accessories(args.output.parent)
