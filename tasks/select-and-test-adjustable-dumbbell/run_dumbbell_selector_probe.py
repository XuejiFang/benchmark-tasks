"""Measure how a physical U-selector lifts one or two nested weight pairs."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


MASSES = {'Handle':1.10,'Light':.70,'Heavy':.90}
SLOTS = {'light':.068,'heavy':.042}


def author(output, level, *, extended_grip=False,compact=False,mass_scale=1.):
    from pxr import Gf, UsdPhysics, UsdShade
    from build_key_return_scene import bar, box, rigid, stage_at
    from build_tea_probe import contact

    stage = stage_at(output,'SelectorDumbbell')
    root = '/SelectorDumbbell'
    steel = (.26,.30,.33)
    grip = (.30,.34,.36)
    handle = root+'/Handle'
    handle_body = rigid(stage,handle,MASSES['Handle']*mass_scale)
    UsdPhysics.MassAPI(handle_body).CreateCenterOfMassAttr(Gf.Vec3f(0.,0.,.044))
    handgrip = bar(stage,handle+'/HandGrip',(0.,-.040,.105),(0.,.040,.105),.014,grip)
    if compact:
        rubber = UsdShade.Material.Define(stage,root+'/HandleRubber')
        material = UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
        material.CreateStaticFrictionAttr(1.25)
        material.CreateDynamicFrictionAttr(1.06)
        UsdShade.MaterialBindingAPI.Apply(handgrip).Bind(rubber,materialPurpose='physics')
    for side,y in (('L',-.028),('R',.028)):
        box(stage,handle+'/Yoke'+side,(0.,y,.056),(.034,.010,.085),steel)
        box(stage,handle+'/Feet'+side,(0.,y,.007),(.073,.017,.014),steel)
    box(stage,handle+'/SelectorHousing',(-.056,0.,.045),(.046,.072,.029),(.29,.34,.37))
    for slot,z in SLOTS.items():
        shelf_gap = .013 if extended_grip else .008
        tine_gap = .011 if extended_grip else .007
        box(stage,handle+'/LowerShelf_'+slot,(-.088,0.,z-shelf_gap),(.030,.097,.007),steel)
        box(stage,handle+'/UpperShelf_'+slot,(-.088,0.,z+shelf_gap+.001),(.030,.097,.007),steel)
        for side,y in (('L',-.044),('R',.044)):
            box(stage,f'{handle}/TineGuideLower_{slot}{side}',(-.026,y,z-tine_gap),(.023,.011,.006),steel)
            box(stage,f'{handle}/TineGuideUpper_{slot}{side}',(-.026,y,z+tine_gap),(.023,.011,.006),steel)
    for name,y,x,z,color in (
        ('Light',.060,.001 if compact else .009,.078,(.28,.45,.55)),
        ('Heavy',.085,.012 if compact else .042,.052,(.65,.40,.25))
    ):
        tier = root+'/'+name
        rigid(stage,tier,MASSES[name]*mass_scale)
        outer = name=='Heavy'
        for side,sign in (('L',-1.),('R',1.)):
            if outer:
                box(stage,tier+'/Plate'+side,(0.,sign*y,.037),(.201,.018,.074),color)
            else:
                cut_start,cut_end = (-.003,.027) if compact else (.027,.057)
                segments = ((-.0725,cut_start,0.,.074),(cut_end,.0725,0.,.074),
                            (cut_start,cut_end,.064,.074))
                for i,(a,b,low,high) in enumerate(segments):
                    box(stage,f'{tier}/Plate{side}{i}',((a+b)/2,sign*y,(low+high)/2),
                        (b-a,.018,high-low),color)
            # The selector still crosses this ledge, while the bare handle feet clear it.
            near = .040
            box(stage,tier+'/LiftTab'+side,(x,sign*(y+near)/2,z),(.024,y-near,.008),color)
        box(stage,tier+'/CrossBrace',(.099 if outer else .079,0.,.020),(.015,2*y,.011),color)
    if level in SLOTS:
        pin = root+'/Selector'
        rigid(stage,pin,.052,(-.088,0.,SLOTS[level]))
        for side,y in (('L',-.044),('R',.044)):
            bar(stage,pin+'/Tine'+side,(0.,y,0.),(.113 if compact else .150,y,0.),.0038,(.66,.68,.70))
        grip_x = -.035 if extended_grip else 0.
        bar(stage,pin+'/Grip',(grip_x,-.045,0.),(grip_x,.045,0.),.0045,(.78,.48,.19))
        if extended_grip:
            grip_face = pin+'/GripFace'
            box(stage,grip_face,(grip_x,0.,.003),(.014,.033,.018),(.78,.48,.19))
            facing = UsdShade.Material.Define(stage,root+'/RubberGrip')
            material = UsdPhysics.MaterialAPI.Apply(facing.GetPrim())
            material.CreateStaticFrictionAttr(1.30)
            material.CreateDynamicFrictionAttr(1.10)
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(grip_face)).Bind(
                facing,materialPurpose='physics')
            for side,y in (('L',-.041),('R',.041)):
                bar(stage,pin+'/GripBridge'+side,(grip_x,y,0.),(0.,y,0.),.0038,(.78,.48,.19))
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            contact(p,.00026)
    stage.GetRootLayer().Save()


def run(output, level, *, extended_grip=False,compact=False):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    import omni.physics.tensors as tensors
    import torch
    from scipy.spatial.transform import Rotation
    from articulated_fixture import array

    usd = output/'dumbbell.usda'
    author(usd,level,extended_grip=extended_grip,compact=compact)
    dt = 1/480
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt,device='cpu',use_fabric=False,physics=PhysxCfg(enable_ccd=True)))
    root = '/World/Dumbbell'
    sim.stage.DefinePrim(root,'Xform').GetReferences().AddReference(str(usd.resolve()))
    floor = sim_utils.CuboidCfg(size=(.52,.42,.020),collision_props=sim_utils.CollisionPropertiesCfg())
    floor.func('/World/Bench',floor,translation=(0.,0.,-.010))
    sim.reset()
    physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
    physics.set_subspace_roots('/')
    names = [*MASSES, *(['Selector'] if level in SLOTS else [])]
    bodies = {name:physics.create_rigid_body_view(root+'/'+name) for name in names}
    if min(view.count for view in bodies.values())!=1:
        raise RuntimeError('Missing a physical handle, weight stack, or installed selector')
    axis = torch.zeros(1,dtype=torch.int32,device=sim.device)
    handle = bodies['Handle']
    rows = []
    total = MASSES['Handle'] + (MASSES['Light']+.052 if level in SLOTS else 0.)
    if level=='heavy':
        total += MASSES['Heavy']
    for i in range(1560):
        time = i*dt
        force = np.zeros(3)
        if time >= .6:
            p = array(handle.get_transforms())[0]
            v = array(handle.get_velocities())[0]
            high = .11*min(1.,(time-.6)/1.3)
            force = np.clip(np.array([-150.*p[0],-150.*p[1],8.9*total+350.*(high-p[2])])-18.*v[:3],
                            [-6.,-6.,0.],[6.,6.,36.])
            tilt = Rotation.from_quat(p[3:]).as_rotvec()
            torques = np.clip(-15.*tilt-1.5*v[3:],-3.,3.)
            handle.apply_forces_and_torques_at_position(
                torch.tensor([force.tolist()],device=sim.device),
                torch.tensor([torques.tolist()],device=sim.device),None,axis,True)
        sim.step(render=False)
        if i%24==23:
            poses = {name:array(view.get_transforms())[0].tolist() for name,view in bodies.items()}
            rows.append({'time_s':(i+1)*dt,'lift_force_n':float(force[2]),
                         'z_m':{name:pose[2] for name,pose in poses.items()},'poses_xyzw':poses})
    (output/'trace.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    check = rows[-10:]
    before = rows[9]['z_m']
    elevated = [name for name in ('Light','Heavy') if min(r['z_m'][name]-before[name] for r in check)>=.045]
    expected = {'light':['Light'],'heavy':['Light','Heavy'],'none':[]}[level]
    return {'scope':'nested physical weights lifted via an inserted U-selector',
            'selector_level':level,'elevated_pairs':elevated,'expected_pairs':expected,
            'handle_rise_mm':1000*(rows[-1]['z_m']['Handle']-before['Handle']),
            'correct_weight_selection':elevated==expected and rows[-1]['z_m']['Handle']-before['Handle']>=.07}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--selector',choices=('light','heavy','none'),required=True)
    parser.add_argument('--extended-grip',action='store_true')
    parser.add_argument('--compact',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,device='cpu').app
    try:
        report = run(args.output,args.selector,extended_grip=args.extended_grip,compact=args.compact)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
