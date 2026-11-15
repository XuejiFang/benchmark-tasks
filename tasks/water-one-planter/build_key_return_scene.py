"""Author colliding key cabinet and two rigid key sets for T02 feasibility trials."""

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def stage_at(path, root):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/' + root).GetPrim())
    return stage


def box(stage, path, xyz, size, color):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def bar(stage, path, start, end, radius, color):
    a, b = Gf.Vec3d(*start), Gf.Vec3d(*end)
    axis = b - a
    shape = UsdGeom.Capsule.Define(stage, path)
    shape.CreateHeightAttr(axis.GetLength())
    shape.CreateRadiusAttr(radius)
    shape.AddTranslateOp().Set((a + b) / 2)
    quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), axis.GetNormalized()).GetQuat()
    shape.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(quat)
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape.GetPrim()


def rigid(stage, path, mass, position=(0, 0, 0)):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    return body.GetPrim()


def blade_guide(stage, path, hook_y, material):
    """Make a visible tapered keeper that receives the hanging blade from above."""
    x, y = .112, hook_y+.0025
    outside = [(-.014,-.018),(.014,-.018),(.014,.018),(-.014,.018)]
    faces = [(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    for section, bottom, top, bottom_gap, top_gap in (
        ('Seat',.114,.124,(.0034,.0062),(.0034,.0062)),
        ('Entry',.124,.141,(.0034,.0062),(.0115,.0145)),
    ):
        for i in range(4):
            j = (i+1)%4
            points = []
            for z, gap in ((bottom,bottom_gap),(top,top_gap)):
                inner = [(-gap[0],-gap[1]),(gap[0],-gap[1]),(gap[0],gap[1]),(-gap[0],gap[1])]
                points.extend(Gf.Vec3f(x+dx,y+dy,z)
                              for dx,dy in (outside[i],outside[j],inner[j],inner[i]))
            shape = UsdGeom.Mesh.Define(stage,f'{path}{section}{i}')
            shape.CreatePointsAttr(points)
            shape.CreateFaceVertexCountsAttr([4]*len(faces))
            shape.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
            shape.CreateSubdivisionSchemeAttr('none')
            shape.CreateDisplayColorAttr([Gf.Vec3f(.16,.18,.20)])
            collider = shape.GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('convexHull')
            collider.AddAppliedSchema('PhysxCollisionAPI')
            collider.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0004)
            collider.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(material,materialPurpose='physics')
    box(stage,path+'Mount',(.138,y,.125),(.024,.036,.018),(.20,.23,.27))


def cabinet(output, lined_hooks=False, guide_blades=False):
    stage = stage_at(output / 'key_cabinet.usda', 'Cabinet')
    root = stage.GetDefaultPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    case = '/Cabinet/Case'
    door = '/Cabinet/Door'
    rigid(stage, case, 4.8)
    hinge = (0, .17, .17)
    rigid(stage, door, 1.25, hinge)
    iron = (.20, .23, .27)
    silver = (.62, .65, .67)
    oak = (.47, .30, .16)
    if lined_hooks or guide_blades:
        liner = UsdShade.Material.Define(stage,'/Cabinet/HookLiner')
        material = UsdPhysics.MaterialAPI.Apply(liner.GetPrim())
        material.CreateStaticFrictionAttr(.66)
        material.CreateDynamicFrictionAttr(.52)
        material.CreateRestitutionAttr(0.)
    box(stage, case+'/Back', (.16, 0, .17), (.018, .36, .34), iron)
    for name, y in [('Left', .17), ('Right', -.17)]:
        box(stage, f'{case}/{name}', (.08, y, .17), (.16, .018, .34), iron)
    for name, z in [('Ceiling', .33), ('Floor', .01)]:
        box(stage, f'{case}/{name}', (.08, 0, z), (.16, .34, .02), iron)
    for name, y in [('Left', .09), ('Right', -.09)]:
        shaft = bar(stage,case+f'/{name}HookShaft',(.151,y,.237),(.078,y,.237),.004,
                    (.28,.28,.30) if lined_hooks else silver)
        tip = bar(stage,case+f'/{name}HookTip',(.078,y,.237),(.070,y,.254),.004,silver)
        if lined_hooks:
            for collider in (shaft,tip):
                collider.AddAppliedSchema('PhysxCollisionAPI')
                collider.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0005)
                collider.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
                UsdShade.MaterialBindingAPI.Apply(collider).Bind(liner,materialPurpose='physics')
        if guide_blades:
            blade_guide(stage,case+f'/{name}Guide',y,liner)
    box(stage, door+'/Panel', (-.009, -.17, 0), (.018, .336, .324), oak)
    for name, z in [('Upper', .041), ('Lower', -.041)]:
        bar(stage, door+f'/Handle{name}', (-.018, -.289, z), (-.058, -.289, z), .005, silver)
    bar(stage, door+'/HandleBar', (-.058, -.289, -.041), (-.058, -.289, .041), .006, silver)
    fixed = UsdPhysics.FixedJoint.Define(stage, '/Cabinet/Mount')
    fixed.CreateBody1Rel().SetTargets([case])
    rotation = UsdPhysics.RevoluteJoint.Define(stage, '/Cabinet/Hinge')
    rotation.CreateBody0Rel().SetTargets([case])
    rotation.CreateBody1Rel().SetTargets([door])
    rotation.CreateLocalPos0Attr(Gf.Vec3f(*hinge))
    rotation.CreateAxisAttr('Z')
    rotation.CreateLowerLimitAttr(-105)
    rotation.CreateUpperLimitAttr(0)
    rotation.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(rotation.GetPrim(), 'angular')
    drive.CreateDampingAttr(.018)
    drive.CreateStiffnessAttr(0)
    stage.GetRootLayer().Save()


def ring_section(stage, path, start, end, color):
    """One segment of round wire with planar ends shared by its neighbors."""
    side_count = 8
    vertices = []
    for angle in (start,end):
        for cross in range(side_count):
            azimuth = math.tau*cross/side_count
            radius = .021+.002*math.cos(azimuth)
            vertices.append(Gf.Vec3f(radius*math.cos(angle),.002*math.sin(azimuth),radius*math.sin(angle)))
    faces = [tuple(reversed(range(side_count))),tuple(range(side_count,side_count*2))]
    for a in range(side_count):
        b = (a+1)%side_count
        faces.append((a,b,b+side_count,a+side_count))
    shape = UsdGeom.Mesh.Define(stage,path)
    shape.CreatePointsAttr(vertices)
    shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
    shape.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    shape.CreateSubdivisionSchemeAttr('none')
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collider = shape.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('convexHull')


def key_set(output, name, color, segment_ring=False):
    stage = stage_at(output / f'{name}.usda', 'KeySet')
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.068)
    metal = (.61, .63, .65)
    # The 37 mm inner aperture is genuine and retains a colliding metal ring.
    if segment_ring:
        count = 24
        for i in range(count):
            ring_section(stage,f'/KeySet/Ring{i:02}',math.tau*i/count,math.tau*(i+1)/count,metal)
    else:
        radius = .021
        ring = [(radius * math.cos(i * math.tau / 16), 0,
                 radius * math.sin(i * math.tau / 16)) for i in range(16)]
        for i, start in enumerate(ring):
            bar(stage,f'/KeySet/Ring{i:02}',start,ring[(i+1)%len(ring)],.002,metal)
    box(stage, '/KeySet/Fob', (0, 0, -.044), (.036, .014, .051), color)
    box(stage, '/KeySet/Blade', (.004, 0, -.085), (.010, .004, .034), metal)
    box(stage, '/KeySet/Tooth', (.010, 0, -.091), (.004, .004, .008), metal)
    stage.GetRootLayer().Save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--lined-hooks',action='store_true')
    parser.add_argument('--segment-ring',action='store_true')
    parser.add_argument('--guide-blades',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cabinet(args.output,args.lined_hooks,args.guide_blades)
    key_set(args.output,'blue_key',(.09,.26,.55),args.segment_ring)
    key_set(args.output,'orange_key',(.87,.31,.07),args.segment_ring)
