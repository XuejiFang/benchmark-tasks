"""Change a physical selector pin and test-lift the resulting adjustable dumbbell."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_dumbbell_trial import INSTRUCTION, STEP_LIMIT, audit

ORIGIN = np.array([.405,-.065,.710])
MOUNTS = {'left':(0.,.305,.75),'right':(0.,-.305,.82)}


def run(output,feedback=False,settle_handle=False,pedestal=False,dock_handle=False,compact=False,
        asset_path=None,dream=False,setup_only=False,clearance_feedback=False,mass_scale=1.):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from PIL import Image
    import torch
    from yam_arena import ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial
    from run_dumbbell_selector_probe import author

    source = asset_path or output/'dumbbell.usda'
    if asset_path is None:
        author(source,'light',extended_grip=True,compact=compact,mass_scale=mass_scale)
    bodies = {name:'/'+name for name in ('Handle','Selector','Light','Heavy')}
    height = .047 if pedestal else 0.
    origin = ORIGIN+np.array([0.,0.,height])
    mounts = MOUNTS | {'left':(0.,.305,.75+height),'right':(0.,-.305,.82+height*.5)}
    station.BASES = mounts.copy()

    def room(*_):
        cradle = (.20,.24,.26)
        room = [scene._box('worktop',(.70,.76,.04),(.39,0.,.69),(.34,.25,.20)),
                scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.42,.43,.42)),
                scene._box('cradle_back',(.014,.216,.034),tuple(origin+[.110,0.,.017]),cradle),
                scene._box('cradle_left',(.217,.012,.029),tuple(origin+[0.,.104,.0145]),cradle),
                scene._box('cradle_right',(.217,.012,.029),tuple(origin+[0.,-.104,.0145]),cradle),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1420.))]
        if pedestal:
            room.append(scene._box('cradle_support',(.242,.235,height),tuple(origin+[0.,0.,-height/2]),cradle))
        return room

    install(source,output,origin,(0.,0.,0.,1.),bodies,room_factory=room,robot_contact_offset=.00045)
    width,height_px = (320,176) if dream else (640,360)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True,cam_height=height_px,cam_width=width,control_hz=20,object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='adjustable-dumbbell-contact',instruction=INSTRUCTION)
        env.reset(specification,seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        eye = [.08,-.18,1.23] if setup_only else [.13,-.55,1.04]
        camera.set_world_poses_from_view(torch.tensor([eye],device=base.device),
                                         torch.tensor([[.40,-.05,.76]],device=base.device))
        if dream or setup_only:
            base.sim.render()
            camera.update(.05,force_recompute=True)
        if setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        observer = MechanismObserver(base.sim.stage)
        robots = {name:base.scene[name+'_robot'] for name in MOUNTS}
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        down = Rotation.from_quat(DOWN_GRIP)
        wrists = {'left':Rotation.from_euler('z',-90.,degrees=True)*down,
                  'right':Rotation.from_euler('x',28.,degrees=True)*Rotation.from_euler('z',90.,degrees=True)*down}
        right_grip_frame = wrists['right']

        with (output/'trace.jsonl').open('w') as log, record_trial(output,True,width=width,height=height_px) as record:
            def step(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('Dumbbell trial exceeded its control limit')
                obs = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()['bodies']
                contacts = {}
                for name in bodies:
                    sensor = base.scene.sensors['mechanism_contact_'+name]
                    forces = array(sensor.data.normal_force_matrix_w).reshape(-1,6,3).sum(0)
                    contacts[name] = np.linalg.norm(forces,axis=1).tolist()
                poses = {name:state[path]['pose_xyzw'] for name,path in bodies.items()}
                row = {'step':len(trace),'phase':phase,'poses':poses,'contacts_n':contacts,
                       'action':action.tolist(),'tcp':{s:tcp_world(robot).tolist() for s,robot in robots.items()}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                record(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%25==0:
                    Image.fromarray(obs.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,poses['Selector'][:3],flush=True)
                if not dream and any(sum(forces)>135. for forces in contacts.values()):
                    raise RuntimeError('Dumbbell contact exceeded 135 N')
                return obs

            if dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,step,lambda _:audit(trace),
                                 trace,action,STEP_LIMIT,[source],early_stop_interval=20)

            def hold(count,phase):
                for _ in range(count):
                    step(phase)

            def move(side,point,seconds,phase,orient=False):
                index = 0 if side=='left' else 7

                def send(q,label):
                    action[index:index+6] = q
                    step(label)

                cartesian_move(lambda:arm_q(robots[side]),send,mounts[side],wrists[side].as_quat(),
                               point,seconds,phase,interpolate_orientation=orient)

            def pose(name):
                latest = np.asarray(trace[-1]['poses'][name])
                return latest[:3],Rotation.from_quat(latest[3:])

            hold(34,'settle both stacks and the inserted light pin')
            start = {name:pose(name)[0] for name in bodies}
            start_handle_rotation = pose('Handle')[1]

            def grasp_handle():
                center,rotation = pose('Handle')
                handle = center+rotation.apply([0.,0. if compact else -.027,.105])-wrists['right'].apply([0.,0.,.009])
                action[13] = .43
                move('right',handle+[0.,-.009,.047],2.5,'approach the handle',True)
                move('right',handle,1.45,'enter handle grasp')
                hold(10,'settle handle fingertips')
                action[13] = 0.
                hold(22,'hold the handle in its base')
                if min(trace[-1]['contacts_n']['Handle'][4:])<=.1:
                    raise RuntimeError('Handle did not make bilateral contact')

            if not dock_handle:
                grasp_handle()

            center,rotation = pose('Selector')
            pin = center+rotation.apply([-.035,0.,.003])+[0.,0.,.004]
            action[6] = .25
            move('left',pin+[0.,0.,.054],2.5,'approach the orange pin',True)
            move('left',pin,1.65,'enter selector grasp')
            hold(10,'settle selector fingertips')
            action[6] = 0.
            hold(20,'grasp orange selector')
            if min(trace[-1]['contacts_n']['Selector'][1:3])<=.1:
                raise RuntimeError('Selector did not make bilateral contact')
            withdrawal = .145 if compact else .191 if feedback else .163
            move('left',pin+[-withdrawal,0.,0.],2.7,'withdraw selector along the guides')
            if pose('Selector')[0][0]>start['Selector'][0]-(.113 if compact else .151):
                raise RuntimeError('The real selector did not leave its guides')
            if settle_handle and not dock_handle:
                action[13] = .55
                hold(20,'release the supported handle to level its feet')
                move('right',tcp_world(robots['right'])+[0.,-.016,.048],1.50,'clear the handle')
                hold(25,'settle the handle on the base')

            lateral_bias = np.zeros(3)

            def align_selector(remaining,seconds,phase):
                center,rotation = pose('Handle')
                slot = center+rotation.apply([-.087,0.,.042])
                pin_center,pin_rotation = pose('Selector')
                tcp,tcp_rotation = ik.tcp_pose(arm_q(robots['left']),mounts['left'])
                error = (rotation*pin_rotation.inv()).as_rotvec()
                length = np.linalg.norm(error)
                correction = Rotation.from_rotvec(error*min(1.,np.deg2rad(10.)/max(length,1e-8)))
                wrists['left'] = correction*Rotation.from_matrix(tcp_rotation)
                axis = rotation.apply([1.,0.,0.])
                offset = correction.apply(tcp-pin_center)
                move('left',slot-axis*remaining+offset+rotation.apply(lateral_bias),seconds,phase,True)
                return slot

            if feedback:
                remaining = withdrawal
                for _ in range(68):
                    slot = align_selector(remaining,.45,'feed the pin through the lower guides')
                    hold(5,'measure pin depth and cross-axis alignment')
                    center,frame = pose('Handle')
                    selector,orientation = pose('Selector')
                    slot = center+frame.apply([-.087,0.,.042])
                    error = frame.inv().apply(slot-selector)
                    angle = (frame*orientation.inv()).magnitude()
                    lateral_bias[1:] = np.clip(lateral_bias[1:]+.64*error[1:],-.021,.021)
                    if np.linalg.norm(error)<=.007 and angle<=np.deg2rad(5.):
                        break
                    if np.max(np.abs(error[1:]))<.0038 and angle<=np.deg2rad(7.):
                        remaining = max(0.,remaining-.009)
                    if min(trace[-1]['contacts_n']['Selector'][1:3])<=.1:
                        raise RuntimeError('Selector grip was lost during insertion')
            else:
                for _ in range(3):
                    align_selector(.165,.55,'align both tines with the lower slot')
                for remaining in (.127,.105,.077,.052,.029,.012,0.,0.,0.):
                    slot = align_selector(remaining,.63,'insert the pin into the lower slot')
            hold(16,'seat both tines')
            if np.linalg.norm(pose('Selector')[0]-slot)>.015:
                raise RuntimeError('Selector did not reach lower slot')
            action[6] = .45
            hold(12,'release the inserted selector')
            move('left',tcp_world(robots['left'])+[0.,0.,.067],1.4,'withdraw left arm')
            if clearance_feedback:
                park_q = np.asarray(READY)
                start_q = arm_q(robots['left']).copy()
                for fraction in np.linspace(0.,1.,51)[1:]:
                    action[:6] = start_q+ik.smoothstep(fraction)*(park_q-start_q)
                    step('park the left arm away from the lift')

            if settle_handle or dock_handle:
                grasp_handle()
            wrist = tcp_world(robots['right'])
            raise_by = .174 if clearance_feedback else .139 if compact else .076
            lift_target = wrist+[0.,0.,raise_by]
            move('right',lift_target,3.3 if compact else 2.45,
                 'test-lift the selected dumbbell')
            if clearance_feedback:
                clear_height = min(pose(name)[0][2]-start[name][2] for name in ('Handle','Light','Heavy'))
                if clear_height<.061:
                    lift_target[2] = wrist[2]+.180
                    move('right',lift_target,.55,'increase the common clearance')
                    hold(6,'measure both lifted stacks')
            hold(36 if compact else 24,'observe the lifted weight plates')
            elevated = {name:min(r['poses'][name][2]-start[name][2] for r in trace[-16:])
                        for name in ('Handle','Light','Heavy')}
            if clearance_feedback:
                landing = lift_target.copy()
                landing[2] = wrist[2]+.157
                move('right',landing,1.35,'lower to the clearance for alignment')
                hold(12,'settle at the alignment height')

                def align_stacks(phase, max_shift):
                    offsets = np.array([start[name]-pose(name)[0] for name in ('Light','Heavy')])
                    _, handle_rotation = pose('Handle')
                    level_error = (start_handle_rotation*handle_rotation.inv()).as_rotvec()
                    magnitude = np.linalg.norm(level_error)
                    rotation_step = level_error*min(1.,np.deg2rad(2.5)/max(magnitude,1e-8))
                    center_error = offsets[:,:2].mean(0)
                    if mass_scale<1. and landing[2]-wrist[2]<=.048:
                        handle_error = (start['Handle']-pose('Handle')[0])[:2]
                        center_error = .48*center_error+.52*handle_error
                    translation = np.r_[np.clip(.64*center_error,-max_shift,max_shift),0.]
                    observed = arm_q(robots['right'])
                    for fraction in (1.,.5,.25,0.):
                        proposed = landing+translation*fraction
                        orientation = Rotation.from_rotvec(rotation_step*fraction)*wrists['right']
                        _, residual, angle = ik.solve(proposed,orientation.as_quat(),observed,
                                                     mounts['right'],iters=180)
                        if residual<=.0005 and angle<=np.deg2rad(.16):
                            break
                    else:
                        raise RuntimeError(f'{phase}: no reachable alignment pose')
                    wrists['right'] = orientation
                    landing[:] = proposed
                    move('right',landing,.85,phase,True)
                    hold(6,'measure the stack centers')

                for _ in range(7):
                    error = max(np.linalg.norm((start[name]-pose(name)[0])[:2])
                                for name in ('Light','Heavy'))
                    tilt = (start_handle_rotation*pose('Handle')[1].inv()).magnitude()
                    if error<.009 and tilt<np.deg2rad(6.):
                        break
                    align_stacks('level and center the stacks above the cradle',.013)
                for height in (.145,.119,.095,.070,.047,.023,.023,.023,0.):
                    landing[2] = wrist[2]+height
                    align_stacks('lower aligned stacks toward the cradle',.005)
            else:
                move('right',wrist,2.45,'put the selected dumbbell back on its base')

            def release_handle():
                action[13] = .65
                hold(14,'open the right gripper')
                move('right',tcp_world(robots['right'])+[0.,-.016,.050],1.35,'clear the handle')
                hold(26,'observe the returned weights')

            release_handle()
            handle_center, handle_frame = pose('Handle')
            stacks_on_base = all(np.linalg.norm(pose(name)[0]-start[name])<=.016
                                 for name in ('Light','Heavy'))
            if (clearance_feedback and mass_scale==1. and stacks_on_base
                    and np.linalg.norm(handle_center-start['Handle'])>.012):
                wrists['right'] = handle_frame*start_handle_rotation.inv()*right_grip_frame
                grasp_handle()
                tcp, rotation = ik.tcp_pose(arm_q(robots['right']),mounts['right'])
                point = np.asarray(tcp)
                for _ in range(10):
                    center, frame = pose('Handle')
                    displacement = start['Handle']-center
                    if np.linalg.norm(displacement)<=.010:
                        break
                    error = (start_handle_rotation*frame.inv()).as_rotvec()
                    norm = np.linalg.norm(error)
                    correction = Rotation.from_rotvec(error*min(1.,np.deg2rad(2.)/max(norm,1e-8)))
                    point += np.clip(.75*displacement,-.005,.006)
                    wrists['right'] = correction*Rotation.from_matrix(rotation)
                    move('right',point,.8,'reseat the handle between the returned stacks',True)
                    hold(8,'check the handle in the cradle')
                    _, rotation = ik.tcp_pose(arm_q(robots['right']),mounts['right'])
                release_handle()
        returned = all(max(np.linalg.norm(np.asarray(r['poses'][name][:3])-start[name]) for r in trace[-20:])<=.012
                       for name in ('Handle','Light','Heavy'))
        retained = max(np.linalg.norm(np.asarray(r['poses']['Selector'][:3])-slot) for r in trace[-20:])<=.012
        clear = max(sum(sum(v) for v in r['contacts_n'].values()) for r in trace[-20:])<=.1
        report = {'scope':'YAM selector transfer and test lift','controls':len(trace),
                  'rise_mm':{name:1000*float(value) for name,value in elevated.items()}}
        if compact and pedestal:
            measured = audit(trace)
            return report | measured | {'complete':measured['score']==1.}
        return report | {'returned':returned,'pin_retained':bool(retained),'released':clear,
                         'complete':bool(min(elevated.values())>=.045 and returned and retained and clear)}
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--feedback',action='store_true')
    parser.add_argument('--settle-handle',action='store_true')
    parser.add_argument('--pedestal',action='store_true')
    parser.add_argument('--dock-handle',action='store_true')
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--clearance-feedback',action='store_true')
    parser.add_argument('--mass-scale',type=float,default=1.)
    parser.add_argument('--asset',type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dream',action='store_true')
    modes.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.dream and args.asset is None:
        parser.error('--dream requires the prebuilt --asset')
    if args.mass_scale<=0.:
        parser.error('--mass-scale must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        final_layout = args.asset is not None
        result = run(args.output,args.feedback or final_layout,args.settle_handle,args.pedestal or final_layout,
                     args.dock_handle or final_layout,args.compact or final_layout,args.asset,args.dream,args.setup_only,
                     args.clearance_feedback,args.mass_scale)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
