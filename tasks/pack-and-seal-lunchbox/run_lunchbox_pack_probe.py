"""Pick up and place the divider, both packages and loose lunchbox lid."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_lunchbox_trial import INSTRUCTION,LID,SIDES,STEP_LIMIT,audit

TUB = '/Container/Geometry/Box'
POSES = {
    'divider': ((.257,.316,.75), .040, 90., .50),
    'wrapped_item_left': ((.181,.252,.75), .032, 0., .45),
    'wrapped_item_right': ((.290,.147,.75), .032, 0., .45),
}


def run(args):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    from PIL import Image
    import torch
    from yam_arena import ik,inspect_embodiment,scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.robot import asset_link_paths
    from yam_arena.runtime import arm_q, tcp_world

    from articulated_fixture import MechanismObserver,array,install
    from expert_motion import cartesian_move
    from record_trial import record_trial
    from run_lunchbox_latch_probe import close_latch

    yaw = Rotation.from_euler('z',-90.,degrees=True)
    fit = json.loads((args.contents/'fit_audit.json').read_text())['placements']
    starts = {name:position for name,(position,*_) in POSES.items()}
    if args.centered:
        starts['wrapped_item_right'] = (.279,.225,.75)

    def room(*_):
        return [scene._box('counter',(.65,.72,.04),(.42,.065,.73),(.38,.3,.24)),
                scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.40,.42,.43)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1400.)),
                *[Object(name=name,object_type=ObjectType.RIGID,
                        usd_path=str((args.contents/f'{name}.usda').resolve()),
                        initial_pose=Pose(position_xyz=position,rotation_xyzw=tuple(map(float,yaw.as_quat()))),
                        spawn_cfg_addon={'activate_contact_sensors':True,'physics_material':scene._grippy(),
                                         'rigid_props':PhysxRigidBodyPropertiesCfg(
                                             solver_position_iteration_count=16,solver_velocity_iteration_count=6)})
                   for name,position in starts.items()]]

    latches = {side:LID+'/Latch_'+side for side in SIDES} if args.latch else {}
    fixture_bodies = {'lid':LID,'tub':TUB,**latches}
    install(args.usd,args.output,(0.,0.,0.),(0.,0.,0.,1.),fixture_bodies,
            room_factory=room,robot_contact_offset=.001)
    original = scene.build_task_environment
    links = asset_link_paths()

    def build(*p,**kw):
        arena = original(*p,**kw)
        configure = arena.env_cfg_callback

        def config(cfg):
            cfg = configure(cfg)
            for name in POSES:
                setattr(cfg.scene,'pack_'+name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+name,
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/'+arm+'Arm'+links[part]
                                           for arm in ('Left','Right')
                                           for part in ('tip_left','tip_right')],history_length=0))
            return cfg

        arena.env_cfg_callback = config
        return arena

    scene.build_task_environment = build
    height,width = (176,320) if args.dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True,cam_height=height,cam_width=width,control_hz=20,object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='pack-lunchbox-contact',instruction=INSTRUCTION if args.latch else
                              'Insert the divider, pack both foods and seat the lid.')
        env.reset(specification,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        robot = base.scene['left_robot']
        mount = (0.,.305,.75)
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.76,-.26,1.10]],device=base.device),
                                         torch.tensor([[.28,.20,.78]],device=base.device))
        observer = MechanismObserver(base.sim.stage)
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        if args.setup_only:
            base.sim.render()
            camera.update(.05,force_recompute=True)
            Image.fromarray(env._observation().images['top_cam']).save(args.output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        with (args.output/'trace.jsonl').open('w') as out,record_trial(
                args.output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=args.max_controls:
                    raise RuntimeError('Lunchbox action limit reached')
                obs = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()
                poses = {name:array(base.scene.rigid_objects[name].data.root_pose_w)[0].tolist()
                         for name in POSES}
                poses.update({name:state['bodies'][path]['pose_xyzw'] for name,path in fixture_bodies.items()})
                contacts = {}
                for name in poses:
                    count = 6 if name in fixture_bodies else 4
                    sensor = ('mechanism_contact_' if count==6 else 'pack_')+name
                    matrix = array(base.scene.sensors[sensor].data.normal_force_matrix_w)
                    contacts[name] = np.linalg.norm(matrix.reshape(-1,count,3).sum(0),axis=1).tolist()
                row = {'phase':phase,'step':len(trace),'action':action.tolist(),'poses':poses,
                       'contacts_n':contacts,'joints':state['articulations'],'observed_q':arm_q(robot).tolist(),
                       'observed_right_q':arm_q(base.scene['right_robot']).tolist()}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                frame(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%30==0:
                    Image.fromarray(obs.images['top_cam']).save(args.output/'latest.png')
                    print('STEP',len(trace),phase,flush=True)
                if not args.dream and (sum(contacts['tub'])>30. or sum(contacts['lid'])>120.
                                       or any(sum(contacts[side])>25. for side in latches)):
                    raise RuntimeError('Lunchbox contact guard exceeded')
                return obs

            if args.dream:
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)
                assets = [args.usd,*(args.contents/f'{name}.usda' for name in starts)]
                return run_dream(args.output,env,specification,env._observation,tick,audit,
                                 trace,action,args.max_controls,assets)

            def hold(n,phase):
                for _ in range(n):
                    tick(phase)

            def send(q,phase):
                action[:6] = q
                tick(phase)

            def relocate(name,offset,goal,roll,width,goal_rotation=None):
                before = np.array(trace[-1]['poses'][name])
                wrist = Rotation.from_euler('z',roll,degrees=True)*Rotation.from_quat(DOWN_GRIP)
                grip = before[:3]+Rotation.from_quat(before[3:]).apply(offset)-wrist.apply([0.,0.,.010])
                height = .896
                hover = np.r_[grip[:2],height]

                def move(point,duration,phase,orient=False):
                    cartesian_move(lambda:arm_q(robot),send,mount,wrist.as_quat(),point,duration,
                                   name+' '+phase,interpolate_orientation=orient)

                action[6] = width
                move([.27,.19,height],1.0 if args.latch else 1.5,'align before approach',True)
                move(hover,1.2 if args.latch else 2.0,'approach')
                move(grip,1.6,'enter grasp')
                action[6] = 0.
                hold(16 if args.latch else 20,name+' grip')
                move(hover,1.3 if args.latch else 1.7,'lift')
                carried = np.array(trace[-1]['poses'][name])
                fingers = trace[-1]['contacts_n'][name][1:3] if name=='lid' else trace[-1]['contacts_n'][name][:2]
                if carried[2]-before[2]<.035 or min(fingers)<.1:
                    raise RuntimeError(f'{name}: no bilateral physical lift')
                tcp = ik.tcp_pose(arm_q(robot),mount)[0]
                grip_offset = tcp-carried[:3]
                if goal_rotation is not None:
                    correction = goal_rotation*Rotation.from_quat(carried[3:]).inv()
                    wrist = correction*wrist
                    grip_offset = correction.apply(grip_offset)
                lift_off = .009 if name=='divider' else .005 if name!='lid' else 0.
                setdown = goal+grip_offset+[0.,0.,lift_off]
                above = np.r_[setdown[:2],height]
                move(above,1.7 if args.latch else 2.8,'carry above tub',goal_rotation is not None)
                if name=='lid' and args.align_lid:
                    target = goal+[0.,0.,.022]
                    for _ in range(6):
                        pose = np.asarray(trace[-1]['poses'][name])
                        error = target-pose[:3]
                        correction = (goal_rotation*Rotation.from_quat(pose[3:]).inv()).as_rotvec()
                        if np.linalg.norm(error)<=.0006 and np.linalg.norm(correction)<=.012:
                            break
                        change = Rotation.from_rotvec(np.clip(.8*correction,-.075,.075))
                        tcp = tcp_world(robot)
                        wrist = change*wrist
                        move(pose[:3]+change.apply(tcp-pose[:3])+np.clip(.85*error,-.012,.012),
                             .45,'align above sealing rim',True)
                        hold(5,'check held lid alignment')
                    pose = np.asarray(trace[-1]['poses'][name])
                    setdown = tcp_world(robot)+goal-pose[:3]+np.r_[np.asarray(args.aim_mm)*.001,0.]
                    above = np.r_[setdown[:2],height]
                move(setdown,2.2,'set down')
                hold(15,name+' hold seating')
                if name=='divider':
                    seated = np.asarray(trace[-1]['poses'][name][:3])
                    correction = goal + [0.,0.,.001] - seated
                    move(tcp_world(robot)+correction,.9,'lower feet into guides')
                    hold(12,'divider settle with feet supported')
                if name=='lid' and args.latch:
                    placed = np.asarray(trace[-1]['poses'][name][:3])
                    correction = np.clip(goal-placed,-.008,.008)
                    move(tcp_world(robot)+correction,.8,'seat aligned lid before releasing')
                    hold(10,'stabilize lid before release')
                action[6] = .28
                hold(16 if args.latch else 20,name+' release')
                move(above,1.1 if args.latch else 1.6,'withdraw')
                action[6] = 1.
                hold(8,name+' clear after release')

            def brace_lid(side='right'):
                right = side=='right'
                arm = base.scene[side+'_robot']
                arm_slice = slice(7,13) if right else slice(6)
                force_slice = slice(3,None) if right else slice(3)
                gripper = 13 if right else 6
                brace_mount = (0.,-.305,.75) if right else mount
                wrist = Rotation.from_quat(DOWN_GRIP).as_quat()

                def send_brace(q,phase):
                    action[arm_slice] = q
                    tick(phase)

                def move(point,seconds,phase,orient=False):
                    cartesian_move(lambda:arm_q(arm),send_brace,brace_mount,wrist,point,seconds,
                                   side+' lid brace '+phase,interpolate_orientation=orient)

                lid = np.asarray(trace[-1]['poses']['lid'][:3])
                xy = lid[:2]+[-.035,-.060 if right else .060]
                hover = np.r_[xy,.895]
                action[gripper] = 0.
                move(hover,2.0,'approach',True)
                target = np.r_[xy,lid[2]+.023]
                move(target,1.6,'lower')
                hold(8,side+' lid brace settle')
                for _ in range(23):
                    if sum(trace[-1]['contacts_n']['lid'][force_slice])>=3.:
                        break
                    target[2] -= .0014
                    move(target,.15,'press down')
                    hold(6,side+' lid brace measure contact')
                else:
                    raise RuntimeError(side+' gripper did not press the lid')
                hold(12,'stabilize brace load')
                for _ in range(8):
                    load = sum(trace[-1]['contacts_n']['lid'][force_slice])
                    if 1.5<=load<=5.5:
                        break
                    target[2] += .0006 if load>5.5 else -.0006
                    move(target,.20,'adjust static brace load')
                    hold(8,'measure settled brace load')
                return lambda:move(hover,1.2,'withdraw')

            hold(45,'reset settling')
            tub = np.array(trace[-1]['poses']['tub'])
            tub_rot = Rotation.from_quat(tub[3:])
            targets = {name:tub[:3]+tub_rot.apply(fit[name]['position_m']) for name in POSES}
            if args.centered:
                targets['wrapped_item_left'] += tub_rot.apply([0.,.008,0.])
            if not args.latches_only:
                for name,(_,handle_z,roll,width) in POSES.items():
                    if args.centered and name=='divider':
                        roll = -90.
                    relocate(name,[0.,0.,handle_z],targets[name],roll,width)
                if not args.contents_only:
                    goal = tub[:3]+tub_rot.apply([0.,0.,.0669])
                    relocate('lid',[-.026,0.,.022],goal,0.,.48,tub_rot if args.latch else None)
                    if args.latch:
                        begin = arm_q(robot).copy()
                        for k in range(1,33):
                            send(begin+ik.smoothstep(k/32)*(np.asarray(READY)-begin),'park left gripper above the worktop')
            release_brace = brace_lid() if args.latch else None
            closure_order = ('back','right','left') if args.after_front else ('back','front','right','left') if latches else ()
            for turn,side in enumerate(closure_order):
                right = side=='right'
                if right:
                    release_brace()
                    release_brace = brace_lid('left') if args.staged_brace else None
                if side=='left' and args.staged_brace:
                    release_brace()
                    right_robot = base.scene['right_robot']
                    ready_from = arm_q(right_robot).copy()
                    action[13] = 1.
                    for k in range(1,33):
                        action[7:13] = ready_from+ik.smoothstep(k/32)*(np.asarray(READY)-ready_from)
                        tick('park right gripper after releasing its latch')
                    release_brace = brace_lid('right')
                arm = base.scene['right_robot'] if right else robot
                arm_joints = slice(7,13) if right else slice(0,6)
                grip = 13 if right else 6
                if turn:
                    begin = arm_q(arm).copy()
                    action[grip] = 1.
                    for k in range(1,29):
                        action[arm_joints] = begin+ik.smoothstep(k/28)*(np.asarray(READY)-begin)
                        tick(side+' latch return to approach posture')
                action[grip] = 0.

                def observe_latch():
                    row = trace[-1]
                    joint = row['joints'][LID]
                    angle = joint['positions'][joint['names'].index('latch_'+side)]
                    return float(np.rad2deg(angle)),np.asarray(row['poses'][side])

                def send_latch(q,phase):
                    action[arm_joints] = q
                    tick(phase)

                close_latch(side,arm,(0.,-.305,.75) if right else mount,observe_latch,send_latch)
                hold(12,side+' latch released')
                if observe_latch()[0]>4.:
                    raise RuntimeError(f'{side} latch stayed open at {observe_latch()[0]:.2f} degrees')
                action[grip] = 1.
            if release_brace is not None:
                release_brace()
            hold(30,'final released settle')
        report = {'scope':'physical packing and optionally lid seating','steps':len(trace),
                'max_placement_error_m':{name:float(max(np.linalg.norm(np.array(row['poses'][name][:3])-target)
                                                       for row in trace[-20:])) for name,target in targets.items()},
                'final_lid_pose_xyzw':trace[-1]['poses']['lid']}
        return report | audit(trace) if args.latch else report
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd',type=Path,required=True)
    parser.add_argument('--contents',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contents-only',action='store_true')
    parser.add_argument('--centered',action='store_true')
    parser.add_argument('--latch',action='store_true')
    parser.add_argument('--align-lid',action='store_true')
    parser.add_argument('--aim-mm',type=float,nargs=2,default=(0.,0.),metavar=('X','Y'))
    parser.add_argument('--staged-brace',action='store_true')
    parser.add_argument('--max-controls',type=int,default=STEP_LIMIT)
    parser.add_argument('--latches-only',action='store_true')
    parser.add_argument('--after-front',action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dream',action='store_true')
    modes.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.dream or args.setup_only:
        args.latch = args.centered = True
    if args.latches_only:
        args.latch = args.centered = True
    if args.after_front and not args.latches_only:
        parser.error('--after-front requires the component-only trial')
    if args.latch and (not args.centered or args.contents_only):
        parser.error('--latch requires --centered and the lid')
    if args.align_lid and (not args.latch or args.latches_only or args.dream):
        parser.error('--align-lid requires a complete expert packing trial')
    if args.staged_brace and (not args.latch or args.dream):
        parser.error('--staged-brace requires an expert latch trial')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        result = run(args)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
