"""Attempt physical robot contact with the monitor height lock and bezel."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_monitor_trial import INSTRUCTION, STEP_LIMIT, audit


ROOT = (.28, .09, .49)
PIN_LOCAL = (-.095, -.026, -.039)
PIN_PUSH_PAD = (-.120, -.051, -.039)
PIN_PATH = '/Links/height_pin'
SCREEN_PATH = '/Links/monitor'
MOUNTS = ((0., .305, .75), (0., -.305, .75))


def build(source, destination, *, push_pin=False, tilt=False,tilt_friction=.033):
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from build_key_return_scene import box, stage_at
    from build_tea_probe import contact
    from run_monitor_balance_probe import add_height_pin

    stage = stage_at(destination, 'MonitorStand')
    root = '/MonitorStand'
    stage.GetDefaultPrim().GetReferences().AddReference(str(source.resolve()))
    for prim in list(Usd.PrimRange(stage.GetDefaultPrim())):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
    height = UsdPhysics.PrismaticJoint.Get(stage, root+'/Joints/height_slide')
    low, high = height.GetLowerLimitAttr().Get(), height.GetUpperLimitAttr().Get()
    midpoint = (low+high)/2
    moving = [stage.GetPrimAtPath(root+suffix) for suffix in (SCREEN_PATH, '/Links/hinge')]
    mass = .021 + sum(UsdPhysics.MassAPI(link).GetMassAttr().Get() for link in moving)
    for link in moving:
        link.GetAttribute('physxRigidBody:disableGravity').Set(False)
    pin = add_height_pin(stage, root, (low, midpoint, high), rail_path=root+'/LockRail')
    box(stage, pin+'/GripExtension', (0., -.016, -.020), (.011, .008, .040), (.37, .38, .40))
    box(stage, pin+'/GripReach', (-.047, -.023, -.039), (.100, .009, .010), (.37, .38, .40))
    box(stage, pin+'/GripTab', PIN_LOCAL, (.024, .011, .023), (.81, .36, .15))
    contact(stage.GetPrimAtPath(pin+'/GripTab'), .0003)
    if push_pin:
        box(stage,pin+'/PushExtension',(-.106,-.037,-.039),(.046,.038,.016),(.30,.32,.35))
        box(stage,pin+'/PushPad',PIN_PUSH_PAD,(.036,.014,.048),(.85,.40,.16))
        contact(stage.GetPrimAtPath(pin+'/PushPad'),.0003)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(moving[0]).GetInverse()
    if push_pin:
        lip = matrix.Transform(Gf.Vec3d(.133,.084,.295))
        for side,dx in (('Left',-.016),('Right',.016)):
            bridge = matrix.Transform(Gf.Vec3d(.133+dx,.028,.295))
            box(stage,root+SCREEN_PATH+'/BezelBridge'+side,bridge,(.010,.116,.014),(.26,.28,.29))
    else:
        lip = matrix.Transform(Gf.Vec3d(.059,-.045,.283))
    box(stage,root+SCREEN_PATH+'/BezelGrip',lip,(.036,.024,.028) if push_pin else (.046,.016,.028),(.24,.25,.27))
    contact(stage.GetPrimAtPath(root+SCREEN_PATH+'/BezelGrip'), .0004)

    spring = UsdPhysics.DriveAPI.Get(height.GetPrim(), 'linear')
    spring.GetStiffnessAttr().Set(55.)
    spring.GetDampingAttr().Set(40.)
    spring.GetTargetPositionAttr().Set(midpoint+mass*9.81/55.)
    hinge = UsdPhysics.RevoluteJoint.Get(stage,root+'/Joints/tilt_hinge')
    if tilt:
        PhysxSchema.PhysxJointAPI.Apply(hinge.GetPrim()).CreateJointFrictionAttr(tilt_friction)
        # The rear counterweight balances the screen about the tilt axis.
        UsdPhysics.MassAPI(moving[0]).CreateCenterOfMassAttr(hinge.GetLocalPos1Attr().Get())
    anchor = UsdPhysics.FixedJoint.Define(stage, root+'/CounterMount')
    anchor.CreateBody1Rel().SetTargets([root+'/Links/base'])
    stage.GetRootLayer().Save()
    return midpoint,np.array(lip),np.array(hinge.GetLocalPos1Attr().Get())


def run(source, output, pin_only, *, push_pin=False, tilt=False,tilt_friction=.033,
        asset_path=None, dream=False, setup_only=False):
    from inspect_robots import Scene
    from isaaclab import sim as sim_utils
    from PIL import Image
    import torch
    from yam_arena import inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    asset = asset_path or output/'stand.usda'
    if asset_path is None:
        midpoint,lip,hinge_local = build(source,asset,push_pin=push_pin,tilt=tilt,tilt_friction=tilt_friction)
    else:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(asset))
        root = str(stage.GetDefaultPrim().GetPath())
        height = UsdPhysics.PrismaticJoint.Get(stage,root+'/Joints/height_slide')
        midpoint = (height.GetLowerLimitAttr().Get()+height.GetUpperLimitAttr().Get())/2
        hinge = UsdPhysics.RevoluteJoint.Get(stage,root+'/Joints/tilt_hinge')
        hinge_local = np.asarray(hinge.GetLocalPos1Attr().Get())
        lip = np.asarray(stage.GetPrimAtPath(root+SCREEN_PATH+'/BezelGrip').GetAttribute('xformOp:translate').Get())
    rotation = Rotation.from_euler('z', 90., degrees=True)
    paths = {'pin': PIN_PATH, 'bezel': SCREEN_PATH}

    def room(*_):
        return [
            scene._box('stand_counter', (.43, .51, .035), (.385, .10, .4725), (.32, .24, .18)),
            scene._box('floor', (3., 3., .04), (.45, 0., -.02), (.39, .40, .39)),
            scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1450.)),
        ]

    install(asset, output, ROOT, rotation.as_quat(), paths, room_factory=room, robot_contact_offset=.001)
    width,height = (320,176) if dream else (640,360)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True, cam_height=height, cam_width=width, control_hz=20, object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        instruction = INSTRUCTION if tilt else 'Lower the monitor to the middle slot and relock it.'
        specification = Scene(id='monitor-height-lock',instruction=instruction)
        env.reset(specification,seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        robots = (base.scene['left_robot'], base.scene['right_robot'])
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.49, -.35, 1.30]], device=base.device),
                                         torch.tensor([[.30, .08, .86]], device=base.device))
        if dream or setup_only:
            base.sim.render()
            camera.update(.05,force_recompute=True)
        if setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        observer = MechanismObserver(base.sim.stage)
        action = np.array([*READY, 1., *READY, 1.])
        trace = []
        wrists = [Rotation.from_euler('z', yaw, degrees=True)*Rotation.from_quat(DOWN_GRIP)
                  for yaw in (-45., 0.)]

        with (output/'trace.jsonl').open('w') as raw, record_trial(output,True,width=width,height=height) as record:
            def step(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('The monitor trial exceeded the control limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()
                joint = next(item for item in state['articulations'].values() if 'height_pin' in item['names'])
                contacts = {}
                for name in paths:
                    sensor = base.scene.sensors['mechanism_contact_'+name]
                    force = array(sensor.data.normal_force_matrix_w).reshape(-1, 6, 3).sum(0)
                    contacts[name] = np.linalg.norm(force, axis=-1).tolist()
                row = {'step': len(trace), 'phase': phase, 'action': action.tolist(),
                       'contacts_n': contacts, 'joint': dict(zip(joint['names'], joint['positions'])),
                       'tcp_xyz_m': [tcp_world(robot).tolist() for robot in robots],
                       'poses': {name: state['bodies'][path]['pose_xyzw'] for name, path in paths.items()}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                raw.write(json.dumps(row, allow_nan=False)+'\n')
                raw.flush()
                record(observation.images['top_cam'])
                if len(trace) % 25 == 0:
                    print('STEP', len(trace), phase, row['joint'], flush=True)
                if not dream and max(sum(force) for force in contacts.values())>155.:
                    raise RuntimeError('Monitor contact load exceeded the probe guard')
                return observation

            if dream:
                from dream_control import run_dream

                source_layers = [Path(layer.realPath) for layer in stage.GetUsedLayers()
                                 if layer.realPath and Path(layer.realPath).resolve()!=asset.resolve()]
                return run_dream(output,env,specification,env._observation,step,audit,trace,action,
                                 STEP_LIMIT,[asset,*source_layers])

            def hold(ticks, phase):
                for _ in range(ticks):
                    step(phase)

            def move(arm, point, seconds, phase, orient=False):
                def send(q, phase):
                    action[arm*7:arm*7+6] = q
                    step(phase)

                cartesian_move(lambda: arm_q(robots[arm]), send, MOUNTS[arm], wrists[arm].as_quat(),
                               np.array(point), seconds, phase, interpolate_orientation=orient)

            def point(name, local):
                pose = trace[-1]['poses'][name]
                return np.asarray(pose[:3])+Rotation.from_quat(pose[3:]).apply(local)

            def grasp(arm, target, name):
                action[arm*7+6] = .79 if name=='bezel' and push_pin else .46
                move(arm, target+[0., 0., .052], 2.8, 'approach '+name, True)
                move(arm, target, 1.6, 'lower over '+name)
                hold(18, 'settle wrist over '+name)
                command = target.copy()
                for _ in range(8):
                    error = target-tcp_world(robots[arm])
                    if np.linalg.norm(error)<=.002:
                        break
                    command += np.clip(.70*error, -.0035, .0035)
                    if np.linalg.norm(command-target)>.027:
                        break
                    move(arm, command, .25, 'align fingers around '+name)
                    hold(6, 'measure settled wrist around '+name)
                action[arm*7+6] = 0.
                hold(20, 'grasp '+name)
                start = arm*3+1
                if min(trace[-1]['contacts_n'][name][start:start+2]) <= .1:
                    raise RuntimeError(f'{name} did not reach both fingertips')

            hold(35, 'let the stand settle')
            if not pin_only:
                if push_pin:
                    wrists[0] = Rotation.from_quat(DOWN_GRIP)
                adjustment = [-.008,.010,.008] if push_pin else [0.,0.,.014]
                grasp(0,point('bezel',lip)+adjustment,'bezel')
            if push_pin:
                wrists[1] = Rotation.from_euler('y',-55.,degrees=True)*Rotation.from_quat(DOWN_GRIP)
                direction = wrists[1].apply([0.,0.,1.])
                pad = point('pin',PIN_PUSH_PAD)
                action[13] = 0.
                move(1,pad-.060*direction,2.8,'approach orange pin paddle',True)
                move(1,pad-.012*direction,1.1,'touch orange pin paddle')
                start_x = pad[0]
                for i in range(24):
                    pad = point('pin',PIN_PUSH_PAD)
                    target = pad-.012*direction
                    target[0] = start_x-.012*direction[0]+min(.043,.0024*(i+1))
                    move(1,target,.35,'press pin toward its guide')
                    hold(4,'measure pin travel')
                    if trace[-1]['joint']['height_pin']<=-.025 and (not pin_only or
                            abs(trace[-1]['joint']['height_slide']-midpoint)<.008):
                        break
            else:
                grasp(1, point('pin', PIN_LOCAL)+[0., 0., .010], 'pin')
                move(1, tcp_world(robots[1])+[.029, 0., 0.], 1.7, 'retract height pin')
            hold(12, 'verify retraction')
            if min(row['joint']['height_pin'] for row in trace[-10:]) > -.021:
                raise RuntimeError('Height pin did not retract far enough to clear the rail')
            if not pin_only:
                for _ in range(17):
                    error = midpoint-trace[-1]['joint']['height_slide']
                    if abs(error)<=.0028:
                        break
                    dz = float(np.clip(error,-.008,.008))
                    for arm in (0, 1):
                        move(arm, tcp_world(robots[arm])+[0., 0., dz], .28,
                             'follow the screen to the center height')
            if push_pin:
                move(1,tcp_world(robots[1])-.048*direction,1.25,'clear the orange pin paddle')
            action[13] = 1.
            hold(20, 'let spring reengage height pin')
            move(1, tcp_world(robots[1])+[0., 0., .035], 1.1, 'withdraw right gripper')
            if tilt:
                current = trace[-1]
                starting_angle = np.rad2deg(current['joint']['tilt_hinge'])
                pivot = point('bezel',hinge_local)
                frame = Rotation.from_quat(current['poses']['bezel'][3:])
                axis = frame.apply([0.,1.,0.])
                forearm = wrists[0]
                offset = tcp_world(robots[0])-pivot
                lead = 4.5
                for _ in range(28):
                    angle = np.rad2deg(trace[-1]['joint']['tilt_hinge'])
                    if abs(angle+10.)<=.8:
                        break
                    desired = float(np.clip(angle+np.sign(-10.-angle)*lead,-14.5,4.8))
                    adjustment = Rotation.from_rotvec(axis*np.deg2rad(desired-starting_angle))
                    wrists[0] = adjustment*forearm
                    move(0,pivot+adjustment.apply(offset),.42,'tilt the screen at the fixed middle height',True)
                    hold(6,'measure tilt at the hinge')
                    observed = np.rad2deg(trace[-1]['joint']['tilt_hinge'])
                    lead = float(np.clip(lead+.60*(1.8-abs(observed-angle)),2.3,14.))
            action[6] = 1.
            hold(20, 'release the screen')
            if not pin_only:
                move(0, tcp_world(robots[0])+[0., 0., .035], 1.1, 'withdraw left gripper')
            hold(25, 'settle the released stand')

        if tilt:
            return audit(trace,midpoint)
        final = trace[-20:]
        result = {'scope': 'YAM physical height pin and screen contact', 'controls': len(trace),
                'retraction_mm': -1000*min(r['joint']['height_pin'] for r in trace),
                'final_height_mm': 1000*final[-1]['joint']['height_slide'],
                'final_height_error_mm': 1000*max(abs(r['joint']['height_slide']-midpoint) for r in final),
                'final_pin_error_mm': -1000*min(r['joint']['height_pin'] for r in final),
                'final_contact_n': max(sum(r['contacts_n']['pin']+r['contacts_n']['bezel']) for r in final)}
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--asset',type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pin-only', action='store_true')
    parser.add_argument('--push-pin',action='store_true')
    parser.add_argument('--tilt',action='store_true')
    parser.add_argument('--tilt-friction',type=float,default=.033)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dream',action='store_true')
    mode.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.asset is not None:
        args.push_pin = args.tilt = True
    if args.tilt and (args.pin_only or not args.push_pin):
        parser.error('--tilt requires the bilateral paddle and handhold sequence')
    if args.dream and args.asset is None:
        parser.error('--dream requires the prebuilt --asset')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        result = run(args.source,args.output,args.pin_only,push_pin=args.push_pin,tilt=args.tilt,
                     tilt_friction=args.tilt_friction,asset_path=args.asset,dream=args.dream,setup_only=args.setup_only)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
