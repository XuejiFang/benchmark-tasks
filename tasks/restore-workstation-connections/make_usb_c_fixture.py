"""Replace the clearance-only noses and mouths in a workstation USD."""

import argparse
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

METAL = (.63, .68, .72)
DARK = (.11, .14, .17)


def collider(prim, *, hull=False):
    UsdPhysics.CollisionAPI.Apply(prim)
    if hull:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        # Keep PhysX's 1 mm default from expanding the submillimetre shell walls.
        prim.AddAppliedSchema('PhysxConvexHullCollisionAPI')
        prim.CreateAttribute('physxConvexHullCollision:minThickness', Sdf.ValueTypeNames.Float).Set(.00002)
    prim.AddAppliedSchema('PhysxCollisionAPI')
    prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.000025)
    prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)


def box(stage, path, centre, size, colour):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.)
    prim.AddTranslateOp().Set(Gf.Vec3d(*centre))
    prim.AddScaleOp().Set(Gf.Vec3f(*size))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    collider(prim.GetPrim())


def strip(stage, path, origin, stations):
    # Every station is one convex quadrilateral in the shell cross section.
    # Cook the small hull around its own origin at millimetre scale.
    points = []
    for x, vertices in stations:
        points.extend(Gf.Vec3f(1000*x, 1000*y, 1000*z) for y, z in vertices)
    last = len(points)-4
    faces = [[3,2,1,0], [last,last+1,last+2,last+3]]
    for start in range(0, last, 4):
        for i in range(4):
            j = (i+1)%4
            faces.append([start+i,start+j,start+4+j,start+4+i])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*origin))
    mesh.AddScaleOp().Set(Gf.Vec3f(.001, .001, .001))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*METAL)])
    collider(mesh.GetPrim(), hull=True)


def shell(stage, path, origin, *, male):
    """A hollow connector shell with an open tongue path and chamfered tip."""
    if male:
        radius, half_flat, thickness = .00120, .002925, .00025
        profile = ((-.00665,0.,0.), (-.00010,0.,0.), (0.,.00002,-.00010))
        layers = [(x,radius-thickness+inner,radius+outer) for x,inner,outer in profile]
    else:
        radius, half_flat, thickness = .00128, .00289, .00032
        profile = ((0.,.00010), (.00010,0.), (.00790,0.))
        layers = [(x,radius+inner,radius+thickness) for x,inner in profile]
    for side in (-1, 1):
        flats = [(x, [(-half_flat,side*r0),(half_flat,side*r0),
                      (half_flat,side*r1),(-half_flat,side*r1)]) for x,r0,r1 in layers]
        strip(stage, f'{path}/Flat{side+1}', origin, flats)
        for segment in range(8):
            a = (-.5+segment/8)*math.pi
            b = a+math.pi/8
            stations = []
            for x,r0,r1 in layers:
                arc = [(side*(half_flat+r*math.cos(t)),r*math.sin(t))
                       for t,r in ((a,r0),(a,r1),(b,r1),(b,r0))]
                stations.append((x, arc))
            strip(stage, f'{path}/Side{side+1}_{segment}', origin, stations)


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    original = Sdf.Layer.FindOrOpen(str(source))
    if original is None:
        raise ValueError('Source USD could not be read')
    revised = Sdf.Layer.CreateNew(str(output))
    revised.TransferContent(original)
    stage = Usd.Stage.Open(revised)
    root = str(stage.GetDefaultPrim().GetPath())
    for device in ('Dock', 'Device'):
        stage.RemovePrim(f'{root}/{device}/StrainReliefShelf')
    # Support the housing from below and leave the rear cord exit unobstructed.
    for stand, y in (('FirstStand', -.290), ('PlugStand', .290)):
        prim = stage.GetPrimAtPath(f'{root}/{stand}')
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(.225, y, .0245))
        prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(.040, .042, .049))
    for name in ('First', 'Second'):
        parent = f'{root}/{name}'
        for child in ('MinusWall', 'PlusWall', 'MinusSkin', 'PlusSkin'):
            if not stage.RemovePrim(f'{parent}/{child}'):
                raise ValueError('Expected a clearance-only source plug')
        shell(stage, f'{parent}/MolexPlug', (0.,0.,0.), male=True)
        grip = stage.GetPrimAtPath(f'{parent}/Grip')
        grip.GetAttribute('xformOp:translate').Set(Gf.Vec3d(-.0177, 0., 0.))
        box(stage, f'{parent}/NoseShoulder',(-.00692,0.,0.),(.00060,.009,.004),DARK)
        box(stage, f'{parent}/StrainRelief',(-.0289,0.,0.),(.002,.006,.005),DARK)
    for parent, tag, y in (('Dock','Primary',-.220),('Dock','Secondary',-.193),('Device','Primary',.220)):
        prefix = f'{root}/{parent}/{tag}'
        for suffix in ('MinusWall','PlusWall','MinusLip','PlusLip','Stop','Tongue'):
            if not stage.RemovePrim(prefix+suffix):
                raise ValueError('Expected a clearance-only source socket')
        shell(stage, prefix+'MolexReceptacle', (.390,y,.053), male=False)
        box(stage,prefix+'Stop',(.397125,y,.053),(.00155,.009,.004),DARK)
        box(stage,prefix+'Tongue',(.39475,y,.053),(.00370,.00669,.00070),DARK)
    revised.Save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    build(options.source, options.output)
