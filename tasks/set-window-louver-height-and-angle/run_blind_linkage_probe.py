"""Check that a physical tilt linkage moves all window-blind slats together."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


SLATS = (.115,.169,.223)
LINK_DISTANCE = .029
LIFT_HEIGHT = .085


def author(destination, linked, carriage_enabled=False, origin=(0.,0.,0.)):
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade
    from build_key_return_scene import bar, box, rigid, stage_at
    from build_tea_probe import contact

    stage = stage_at(destination,'Blind')
    root = '/Blind'
    rail = root+'/TiltLink'
    aluminum = (.55,.60,.61)
    case = (.42,.46,.45)
    box(stage,root+'/TopRail',(0.,0.,.361 if carriage_enabled else .277),(.095,.376 if carriage_enabled else .356,.026),case)
    for end,y in (('Left',-.173),('Right',.173)):
        box(stage,root+'/Guide'+end,(0.,y,.198 if carriage_enabled else .163),(.035,.020,.303 if carriage_enabled else .230),case)
    carriage = root+'/Carriage'

    if carriage_enabled:
        mark = root+'/HeightStripe'
        box(stage,mark,(-.021,.183,.064+LIFT_HEIGHT),(.008,.017,.008),(.22,.45,.66))
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(mark)).CreateCollisionEnabledAttr(False)
        track = root+'/Track'
        rigid(stage,track,.60,(0.,0.,0.))
        box(stage,track+'/Housing',(.128,.105,.202),(.029,.053,.053),case)
        anchor = UsdPhysics.FixedJoint.Define(stage,root+'/TrackAnchor')
        anchor.CreateBody1Rel().SetTargets([track])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(*origin))
        anchor.CreateLocalPos1Attr(Gf.Vec3f(0.))
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        rigid(stage,carriage,.15,(0.,0.,0.))
        box(stage,carriage+'/BottomRail',(0.,0.,.064),(.044,.308,.018),case)
        for side,y in (('Left',-.155),('Right',.155)):
            box(stage,carriage+'/Slider'+side,(0.,y,.163),(.023,.011,.201),(.35,.37,.36))
        box(stage,carriage+'/LiftWeld',(-.026,.120,.064),(.026,.026,.016),(.29,.33,.34))
        box(stage,carriage+'/LiftSupport',(-.044,.151,.064),(.030,.067,.016),(.29,.33,.34))
        box(stage,carriage+'/LiftExtension',(-.076,.169,.064),(.113,.021,.016),(.29,.33,.34))
        box(stage,carriage+'/LiftTab',(-.131,.169,.064),(.038,.035,.022),(.18,.37,.42))
        slide = UsdPhysics.PrismaticJoint.Define(stage,root+'/Lift')
        slide.CreateAxisAttr('Z')
        slide.CreateBody0Rel().SetTargets([track])
        slide.CreateBody1Rel().SetTargets([carriage])
        slide.CreateCollisionEnabledAttr(False)
        slide.CreateLocalPos0Attr(Gf.Vec3f(0.))
        slide.CreateLocalPos1Attr(Gf.Vec3f(0.))
        slide.CreateLowerLimitAttr(0.)
        slide.CreateUpperLimitAttr(.108)
        plate = carriage+'/BrakePlate'
        box(stage,plate,(.067,.105,.149),(.012,.032,.240),(.30,.35,.35))
        brake = root+'/Brake'
        brake_origin = np.array([.097,.105,.202])
        rigid(stage,brake,.030,brake_origin)
        pad = brake+'/Pad'
        box(stage,pad,(0.,0.,0.),(.018,.034,.030),(.21,.24,.22))
        shoe = UsdPhysics.PrismaticJoint.Define(stage,root+'/BrakeSpring')
        shoe.CreateAxisAttr('X')
        shoe.CreateBody1Rel().SetTargets([brake])
        shoe.CreateLocalPos0Attr(Gf.Vec3f(*(brake_origin+origin)))
        shoe.CreateLocalPos1Attr(Gf.Vec3f(0.))
        shoe.CreateLowerLimitAttr(-.030)
        shoe.CreateUpperLimitAttr(0.)
        preload = UsdPhysics.DriveAPI.Apply(shoe.GetPrim(),'linear')
        preload.CreateStiffnessAttr(650.)
        preload.CreateDampingAttr(6.)
        preload.CreateTargetPositionAttr(-.026)
        lining = UsdShade.Material.Define(stage,root+'/BrakeLining')
        rubber = UsdPhysics.MaterialAPI.Apply(lining.GetPrim())
        rubber.CreateStaticFrictionAttr(.85)
        rubber.CreateDynamicFrictionAttr(.70)
        for path in (plate,pad):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(lining,materialPurpose='physics')
    if linked:
        rigid(stage,rail,.014,(LINK_DISTANCE,.116,SLATS[1]))
        box(stage,rail+'/Tie',(0.,0.,0.),(.006,.007,.132),(.33,.40,.40))
    for i,z in enumerate(SLATS):
        name = root+f'/Slat{i}'
        slat = rigid(stage,name,.022,(0.,0.,z))
        if carriage_enabled:
            # A counterweight along each leading edge balances the moving link.
            offset = -.014*LINK_DISTANCE/(.022*len(SLATS)) if linked else 0.
            UsdPhysics.MassAPI(slat).CreateCenterOfMassAttr(Gf.Vec3f(offset,0.,0.))
        box(stage,name+'/Blade',(0.,0.,0.),(.075,.315,.003),aluminum)
        box(stage,name+'/Crank',(LINK_DISTANCE*.5,.116,0.),(LINK_DISTANCE,.009,.006),case)
        hinge = UsdPhysics.RevoluteJoint.Define(stage,root+f'/Tilt{i}')
        hinge.CreateAxisAttr('Y')
        if carriage_enabled:
            hinge.CreateBody0Rel().SetTargets([carriage])
            hinge.CreateCollisionEnabledAttr(False)
        hinge.CreateBody1Rel().SetTargets([name])
        hinge.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,z))
        hinge.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
        hinge.CreateLowerLimitAttr(-61.)
        hinge.CreateUpperLimitAttr(61.)
        drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular')
        drive.CreateDampingAttr(.0013)
        drive.CreateStiffnessAttr(0.)
        if carriage_enabled:
            PhysxSchema.PhysxJointAPI.Apply(hinge.GetPrim()).CreateJointFrictionAttr(.16)
            if i==1:
                bar(stage,name+'/TiltLever',(0.,-.078,0.),(-.105,-.078,0.),.007,(.48,.41,.26))
                box(stage,name+'/TiltTab',(-.110,-.078,0.),(.037,.041,.016),(.81,.45,.15))
        if linked:
            pivot = UsdPhysics.RevoluteJoint.Define(stage,root+f'/LinkPivot{i}')
            pivot.CreateAxisAttr('Y')
            pivot.CreateBody0Rel().SetTargets([name])
            pivot.CreateBody1Rel().SetTargets([rail])
            pivot.CreateLocalPos0Attr(Gf.Vec3f(LINK_DISTANCE,.116,0.))
            pivot.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,z-SLATS[1]))
            pivot.CreateCollisionEnabledAttr(False)
            if carriage_enabled:
                pivot.CreateExcludeFromArticulationAttr(True)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            contact(prim,.00024)
    stage.GetRootLayer().Save()


def run(output, linked, carriage_enabled=False, stay_still=False):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    import omni.physics.tensors as tensors
    from scipy.spatial.transform import Rotation
    import torch
    from articulated_fixture import array

    asset = output/'window_blind.usda'
    author(asset,linked,carriage_enabled)
    dt = 1/480
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt,device='cpu',use_fabric=False,physics=PhysxCfg(enable_ccd=True)))
    root = '/World/Blind'
    sim.stage.DefinePrim(root,'Xform').GetReferences().AddReference(str(asset.resolve()))
    sim.reset()
    physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
    physics.set_subspace_roots('/')
    slats = [physics.create_rigid_body_view(root+f'/Slat{i}') for i in range(len(SLATS))]
    if min(slat.count for slat in slats)!=1:
        raise RuntimeError('Missing at least one physical blind slat')
    index = torch.zeros(1,dtype=torch.int32,device=sim.device)
    none = torch.zeros((1,3),device=sim.device)
    if carriage_enabled:
        carriage = physics.create_rigid_body_view(root+'/Carriage')
        if carriage.count!=1:
            raise RuntimeError('The blind has no live lifting carriage')

    def angle(slat):
        pose = array(slat.get_transforms())[0]
        return float(Rotation.from_quat(pose[3:]).as_euler('xyz',degrees=True)[1])

    rows = []
    for tick in range(3840):
        time = tick*dt
        target = 0. if stay_still or time<.35 else 47. if time<4.15 else -28.
        current = angle(slats[-1])
        rate = array(slats[-1].get_velocities())[0,4]
        gain,limit = (.030,.24) if carriage_enabled else (.018,.16)
        torque = float(np.clip(gain*(target-current)-.016*rate,-limit,limit))
        if not stay_still and (not carriage_enabled or time<6.75):
            slats[-1].apply_forces_and_torques_at_position(
                none,torch.tensor([[0.,torque,0.]],device=sim.device),None,index,True)
        if carriage_enabled and not stay_still and 1.6<=time<3.4:
            z = array(carriage.get_transforms())[0,2]
            rate = array(carriage.get_velocities())[0,2]
            force = float(np.clip(5.+1200.*(LIFT_HEIGHT-z)-36.*rate,-8.,12.))
            carriage.apply_forces_and_torques_at_position(
                torch.tensor([[0.,0.,force]],device=sim.device),none,None,index,True)
        sim.step(render=False)
        if tick%24==23:
            row = {'time_s':(tick+1)*dt,'desired_top_degrees':target,
                   'slat_degrees':[angle(slat) for slat in slats],'drive_torque_nm':torque}
            if carriage_enabled:
                row['height_m'] = float(array(carriage.get_transforms())[0,2])
            rows.append(row)
    (output/'trace.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    windows = ([r for r in rows if 3.6<=r['time_s']<=4.1],rows[-10:])
    max_error = max(abs(v-r['desired_top_degrees']) for sample in windows for r in sample for v in r['slat_degrees'])
    report = {'scope':'physical linkage actuated only through the top slat during the probe',
              'linked':linked,'max_target_error_deg':max_error,
              'final_slat_degrees':rows[-1]['slat_degrees'],'synchronized':max_error<=5.}
    if carriage_enabled:
        height = 0. if stay_still else LIFT_HEIGHT
        report.update(lift_m=rows[-1]['height_m'],height_error_max_m=max(abs(r['height_m']-height) for r in rows[-20:]),
                      passive_control=stay_still,scope='linked slats tilt independently of the released lifting carriage')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--unlinked',action='store_true')
    parser.add_argument('--carriage',action='store_true')
    parser.add_argument('--stay-still',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,device='cpu').app
    try:
        report = run(args.output,not args.unlinked,args.carriage,args.stay_still)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
