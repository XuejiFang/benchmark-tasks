"""Lift, pour, and return a free pitcher using the two YAM grippers."""

import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_watering_trial import DROP_COUNT, INSTRUCTION, ORIGIN, POTS, STEP_LIMIT, audit, classify


MOUNTS = {'left':(0.,.285,.60),'right':(0.,-.285,.60)}
PITCHER = '/Pitcher'
LIP = np.array([.032,0.,.062])


def build(destination):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    from build_key_return_scene import bar, box, rigid, stage_at
    from build_tea_probe import contact

    stage = stage_at(destination,'Watering')
    root = '/Watering'
    pitcher = root+PITCHER
    body = rigid(stage,pitcher,.270,(0.,0.,.0004))
    body.AddAppliedSchema('PhysxRigidBodyAPI')
    body.CreateAttribute('physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool).Set(True)
    body.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(24)
    body.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(8)
    teal = (.10,.35,.38)
    box(stage,pitcher+'/Base',(0.,0.,.002),(.069,.069,.004),teal)
    for sign,label in ((1,'Left'),(-1,'Right')):
        box(stage,pitcher+'/'+label+'X',(sign*.032,0.,.032),(.004,.068,.060),teal)
        box(stage,pitcher+'/'+label+'Y',(0.,sign*.032,.032),(.060,.004,.060),teal)
        box(stage,pitcher+'/'+label+'GripSupport',(-.013,sign*.032,.060),(.019,.010,.035),teal)
        box(stage,pitcher+'/'+label+'Grip',(-.013,sign*.047,.080),(.018,.035,.012),(.22,.28,.26))
        box(stage,pitcher+'/'+label+'SpoutSide',(.017,sign*.027,.061),(.037,.014,.005),teal)
    box(stage,pitcher+'/SpoutDeck',(-.020,0.,.061),(.033,.067,.005),teal)
    rubber = UsdShade.Material.Define(stage,root+'/GripRubber')
    material = UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    material.CreateStaticFrictionAttr(.90)
    material.CreateDynamicFrictionAttr(.70)
    for part in stage.GetPrimAtPath(pitcher).GetChildren():
        contact(part,.0006)
        if part.GetName().endswith('Grip'):
            UsdShade.MaterialBindingAPI.Apply(part).Bind(rubber,materialPurpose='physics')
    for name,(x,y) in POTS.items():
        path = root+'/Planters/'+name
        tone = (.55,.25,.14) if name=='target' else (.22,.43,.30)
        box(stage,path+'/Bottom',(x,y,.002),(.111,.109,.004),tone)
        for sign,tag in ((-1,'Near'),(1,'Far')):
            box(stage,path+'/X'+tag,(x+sign*.053,y,.0375),(.006,.110,.067),tone)
            box(stage,path+'/Y'+tag,(x,y+sign*.052,.0375),(.103,.006,.067),tone)
        for i in range(5):
            th = 2*math.pi*i/5
            foot = np.array([x+.039,y+.026,.004]); top = foot+[.014*math.cos(th),.024*math.sin(th),.107+.009*(i%2)]
            stem = bar(stage,f'{path}/Stem{i}',foot,top,.0017,(.16,.33,.16))
            leaves = UsdGeom.Sphere.Define(stage,f'{path}/Leaf{i}')
            leaves.CreateRadiusAttr(1.)
            leaves.AddTranslateOp().Set(Gf.Vec3d(*top))
            leaves.AddScaleOp().Set(Gf.Vec3f(.010,.014,.007))
            leaves.CreateDisplayColorAttr([Gf.Vec3f(.13,.37+.025*i,.12)])
            UsdPhysics.CollisionAPI(stem).CreateCollisionEnabledAttr(False)
    for i in range(DROP_COUNT):
        x,y,z = i%7,i//7%7,i//49
        path = root+f'/Water/D{i:03d}'
        rigid(stage,path,.000074,((x-3)*.006,(y-3)*.006,.012+z*.006))
        sphere = UsdGeom.Sphere.Define(stage,path+'/Drop')
        sphere.CreateRadiusAttr(.0026)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(.27,.61,.89)])
        contact(sphere.GetPrim(),.00029)
    UsdGeom.Cylinder.Define(stage,root+'/Coaster').CreateRadiusAttr(.056)
    coaster = UsdGeom.Cylinder.Get(stage,root+'/Coaster')
    coaster.CreateHeightAttr(.001)
    coaster.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.0005))
    coaster.CreateDisplayColorAttr([Gf.Vec3f(.56,.46,.33)])
    stage.GetRootLayer().Save()


def run(args):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from inspect_robots import Scene
    import omni.physics.tensors as tensors
    from PIL import Image
    import torch
    from yam_arena import ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.runtime import arm_q
    from articulated_fixture import array, install
    from record_trial import record_trial

    output = args.output
    asset = args.asset or output/'watering.usda'
    if args.asset is None:
        build(asset)
    station.BASES = MOUNTS.copy()

    def room(*_):
        return [scene._box('worktop',(.65,.72,.04),(.415,.030,.62),(.48,.42,.36)),
                scene._box('floor',(3.,3.,.04),(.45,0.,-.020),(.29,.32,.34)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1450.))]

    install(asset,output,ORIGIN,(0.,0.,0.,1.),{'pitcher':PITCHER},room_factory=room,robot_contact_offset=.00072)
    configure_room = scene.build_task_environment

    def configure(*a,**kw):
        arena = configure_room(*a,**kw)
        previous = arena.env_cfg_callback

        def callback(cfg):
            cfg = previous(cfg)
            cfg.sim.dt = 1/480
            cfg.decimation = cfg.sim.render_interval = 24
            cfg.sim.use_fabric = False
            cfg.sim.physics = PhysxCfg(enable_ccd=True)
            return cfg

        arena.env_cfg_callback = callback
        return arena

    scene.build_task_environment = configure
    resolution = (320,176) if args.dream else (640,360)
    env = inspect_embodiment.YamArenaEmbodiment(
        device='cpu',headless=True,cam_width=resolution[0],cam_height=resolution[1],control_hz=20,
        object_jitter=0.,task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='selective-planter-watering',instruction=INSTRUCTION)
        env.reset(specification,seed=0)
        env._task_step = lambda:SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.70,-.49,1.15]],device=base.device),
                                         torch.tensor([[.37,.045,.71]],device=base.device))
        base.sim.render()
        camera.update(.05,force_recompute=True)
        if args.setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
        physics.set_subspace_roots('/')
        root = '/World/envs/env_0/mechanism'
        pitcher = physics.create_rigid_body_view(root+PITCHER)
        drops = physics.create_rigid_body_view(root+'/Water/D*')
        if pitcher.count!=1 or drops.count!=DROP_COUNT:
            raise RuntimeError('Expected one live pitcher and 196 independently simulated drops')
        robots = {side:base.scene[side+'_robot'] for side in MOUNTS}
        action = np.array([*READY,1.,*READY,1.])
        trace = []

        with (output/'trace.jsonl').open('w') as log, record_trial(output,True,width=resolution[0],height=resolution[1]) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('Watering exceeded its control limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                points = array(drops.get_transforms())[:,:3]
                pose = array(pitcher.get_transforms())[0]
                sensor = base.scene.sensors['mechanism_contact_pitcher']
                values = array(sensor.data.normal_force_matrix_w).reshape(-1,6,3).sum(0)
                contact_n = np.linalg.norm(values,axis=1).tolist()
                row = {'step':len(trace),'phase':phase,'pitcher_pose_xyzw':pose.tolist(),
                       'counts':classify(points,pose),'contact_n':contact_n,'action':action.tolist()}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                frame(observation.images['top_cam'])
                if len(trace)%25==0:
                    Image.fromarray(observation.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,row['counts'],np.round(contact_n,1),flush=True)
                return observation

            if args.dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,tick,audit,trace,action,STEP_LIMIT,[asset])

            def hold(count,phase):
                for _ in range(count):
                    tick(phase)

            def joints():
                return {side:np.copy(arm_q(robot)) for side,robot in robots.items()}

            def solve(points,rotations,seed,phase):
                result = {}
                for side in MOUNTS:
                    q,pe,re = ik.solve(points[side],rotations[side].as_quat(),seed[side],MOUNTS[side],iters=170)
                    if pe>.001 or re>math.radians(.3):
                        raise RuntimeError(f'{side} {phase} IK failed: {pe:.4f} m, {math.degrees(re):.2f} deg')
                    result[side] = q
                return result

            def send(target,phase):
                for i,side in enumerate(MOUNTS):
                    action[i*7:i*7+6] = target[side]
                tick(phase)

            def move(points,rotations,seconds,phase):
                seed = joints()
                start = {side:ik.tcp_pose(q,MOUNTS[side])[0] for side,q in seed.items()}
                for alpha in np.linspace(0.,1.,round(seconds*20)+1)[1:]:
                    f = ik.smoothstep(alpha)
                    target = {side:start[side]+f*(points[side]-start[side]) for side in MOUNTS}
                    seed = solve(target,rotations,seed,phase)
                    send(seed,phase)

            def park(target,phase,seconds=2.5):
                start = joints()
                for alpha in np.linspace(0.,1.,round(seconds*20)+1)[1:]:
                    f = ik.smoothstep(alpha)
                    send({side:start[side]+f*(target[side]-start[side]) for side in MOUNTS},phase)

            hold(40,'settle the filled pitcher')
            start = np.array(trace[-1]['pitcher_pose_xyzw'])
            if trace[-1]['counts']['pitcher']!=DROP_COUNT:
                raise RuntimeError('The reset pitcher does not contain its full dose')
            home_rotation = Rotation.from_quat(start[3:])
            grips,rotations,entry = {},{},{}
            for side,sign,roll in (('left',1.,0.),('right',-1.,180.)):
                grips[side] = start[:3]+home_rotation.apply([-.013,sign*.058,.080])
                rotations[side] = home_rotation*Rotation.from_euler('z',-90.*sign,degrees=True)*Rotation.from_euler('y',90.,degrees=True)*Rotation.from_euler('z',roll,degrees=True)
                entry[side] = grips[side]+[0.,sign*.058,0.]
            hover = {side:point+[0.,0.,.075] for side,point in entry.items()}
            action[[6,13]] = .64
            park(solve(hover,rotations,joints(),'overhead approach'),'align both wrists')
            move(entry,rotations,1.6,'descend beside the pitcher')
            move(grips,rotations,1.6,'reach the two grip tabs')
            for opening in np.linspace(.64,0.,24):
                action[[6,13]] = opening
                tick('grasp both tabs')
            hold(14,'settle both grips')
            lift = {side:ik.tcp_pose(q,MOUNTS[side])[0]+[0.,0.,.075] for side,q in joints().items()}
            move(lift,rotations,2.1,'lift the filled pitcher')
            hold(20,'check the upright lift')
            raised = np.array(trace[-1]['pitcher_pose_xyzw'])
            if raised[2]-start[2]<.059 or min(np.take(trace[-1]['contact_n'],[1,2,4,5]))<=.05:
                raise RuntimeError('The pitcher was not lifted under bilateral contact')

            inverse = Rotation.from_quat(raised[3:]).inv()
            offsets,relative = {},{}
            for side,q in joints().items():
                point,rotation = ik.tcp_pose(q,MOUNTS[side])
                offsets[side] = inverse.apply(point-raised[:3])
                relative[side] = inverse*Rotation.from_matrix(rotation)

            def targets(center,rotation):
                return ({side:center+rotation.apply(offsets[side]) for side in MOUNTS},
                        {side:rotation*relative[side] for side in MOUNTS})

            mouth = ORIGIN+[.155,0.,.162]
            move(*targets(mouth-LIP,Rotation.identity()),2.7,'approach the terracotta planter')

            def tip(parameters,phase,stop_at_reserve=False):
                seed = joints()
                for f in parameters:
                    rotation = Rotation.from_euler('y',115.*f,degrees=True)
                    lip = mouth+[.008*f,0.,-.035*f]
                    points,wrists = targets(lip-rotation.apply(LIP),rotation)
                    seed = solve(points,wrists,seed,phase)
                    send(seed,phase)
                    if stop_at_reserve and trace[-1]['counts']['pitcher']<=119:
                        return f
                return f

            approach_angles = .72*np.array([ik.smoothstep(v) for v in np.linspace(0.,1.,81)[1:]])
            release_angles = np.r_[approach_angles,np.linspace(.72,1.,155)[1:],np.ones(40)]
            endpoint = tip(release_angles,'pour one dose into the terracotta planter',True)
            remaining_angles = endpoint*np.square(1.-np.linspace(0.,1.,75)[1:])
            tip(remaining_angles,'stop the flow and retain the remaining dose')
            move(*targets(start[:3]+[0.,0.,.080],home_rotation),2.5,'bring the pitcher to its coaster')
            move(*targets(start[:3]+[0.,0.,.001],home_rotation),2.1,'set the pitcher down')
            action[[6,13]] = 1.
            hold(22,'release the pitcher')
            move(entry,rotations,1.5,'retreat beside the pitcher')
            move(hover,rotations,1.6,'raise both wrists')
            hold(25,'observe the returned pitcher and watered planter')
        return audit(trace)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--asset',type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dream',action='store_true')
    modes.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,device='cpu',enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment

        inspect_embodiment._APP = app
        report = run(args)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
