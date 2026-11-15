"""Check a physical bimanual pinch and lift of the filled pantry carton."""

import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_refill_trial import INSTRUCTION, LID_SEAT, ORIGIN, RUN_LIMIT, audit


def build_lid(path):
    from pxr import UsdPhysics

    from build_key_return_scene import box, stage_at
    from build_tea_probe import cylinder

    stage = stage_at(path,'StorageJarLid')
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.045)
    green = (.27,.59,.53)
    cylinder(stage,'/StorageJarLid/Locator',(0.,0.,.003),.048,.006,green)
    cylinder(stage,'/StorageJarLid/Cap',(0.,0.,.0095),.056,.007,green)
    box(stage,'/StorageJarLid/TabSupport',(-.055,0.,.010),(.036,.028,.008),green)
    box(stage,'/StorageJarLid/Grip',(-.065,0.,.022),(.017,.024,.020),(.20,.37,.32))
    stage.GetRootLayer().Save()


def run(args):
    import isaaclab.sim as sim_utils
    import omni.physics.tensors as tensors
    import torch
    from inspect_robots import Scene
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    from PIL import Image
    from yam_arena import embodiment, ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.robot import asset_link_paths
    from yam_arena.runtime import arm_q

    from articulated_fixture import array
    from expert_motion import cartesian_move
    from record_trial import record_trial

    output,pour_trace,with_lid = args.output,args.pour_trace,args.with_lid
    origin = ORIGIN
    mounts = {'left': (0., .285, .60), 'right': (0., -.285, .60)}
    station.BASES = mounts
    asset_root = args.assets if args.assets else output
    pantry = asset_root/'pantry.usda'
    lid_seat = LID_SEAT
    lid_park = np.array([.255,.20,.6401])
    lid_path = asset_root/'lid.usda'
    if args.assets is None:
        from run_refill_flow_probe import build_layout

        build_layout(pantry,lined=True,round_grain=True,graspable=True)
        if with_lid:
            build_lid(lid_path)
    links = asset_link_paths()

    def room(*_):
        parts = [
            scene._box('worktop', (.62,.62,.04), (.40,0.,.62), (.36,.27,.21)),
            scene._box('floor', (3.,3.,.04), (.45,0.,-.02), (.36,.39,.42)),
            scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1300.)),
            Object(name='pantry', object_type=ObjectType.BASE, usd_path=str(pantry.resolve()),
                   initial_pose=Pose(position_xyz=tuple(origin),rotation_xyzw=(0.,0.,0.,1.))),
        ]
        if with_lid:
            parts.append(Object(
                name='jar_lid',object_type=ObjectType.RIGID,usd_path=str(lid_path.resolve()),
                initial_pose=Pose(position_xyz=tuple(map(float,lid_seat)),rotation_xyzw=(0.,0.,0.,1.)),
                spawn_cfg_addon={'physics_material':scene._grippy(),'activate_contact_sensors':True,
                                 'rigid_props':PhysxRigidBodyPropertiesCfg(kinematic_enabled=False,
                                     solver_position_iteration_count=24,solver_velocity_iteration_count=8)}))
        return parts

    original = scene.build_task_environment

    def build(*args, **kwargs):
        arena = original(*args, **kwargs)
        configure = arena.env_cfg_callback

        def config(cfg):
            cfg = configure(cfg)
            cfg.sim.dt = 1/480
            cfg.decimation = cfg.sim.render_interval = 24
            cfg.sim.use_fabric = False
            cfg.sim.physics = PhysxCfg(enable_ccd=True)
            cfg.scene.carton_contact = ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/pantry/Carton',
                filter_prim_paths_expr=[
                    '{ENV_REGEX_NS}/'+arm+'Arm'+links[tip]
                    for arm in ('Left','Right') for tip in ('tip_left','tip_right')
                ],
                history_length=0)
            if with_lid:
                cfg.scene.lid_contact = ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/jar_lid',
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/LeftArm'+links[tip]
                                            for tip in ('tip_left','tip_right')],
                    history_length=0)
            return cfg

        arena.env_cfg_callback = config
        return arena

    scene.task_room = room
    scene.build_task_environment = build
    bimanual = embodiment.BimanualYamEmbodiment
    embodiment.BimanualYamEmbodiment = lambda *a,**kw: bimanual(*a,**(kw | {'robot_contact_offset':.001}))
    height,width = (176,320) if args.dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        device='cpu',headless=True,cam_height=height,cam_width=width,control_hz=20,object_jitter=0,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='refill-carton',instruction=INSTRUCTION if with_lid else
                              'Lift the filled carton with both grippers.')
        env.reset(specification,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(
            torch.tensor([[.85,-.28,1.20]],device=base.device),
            torch.tensor([[.34,.04,.72]],device=base.device))
        physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
        physics.set_subspace_roots('/')
        carton = physics.create_rigid_body_view('/World/envs/env_0/pantry/Carton')
        grain = physics.create_rigid_body_view('/World/envs/env_0/pantry/Leaf*')
        if carton.count!=1 or grain.count!=80:
            raise RuntimeError(f'Expected one dynamic carton and 80 grains, got {carton.count} and {grain.count}')
        robots = {side:base.scene[side+'_robot'] for side in mounts}
        lid = base.scene.rigid_objects['jar_lid'].data if with_lid else None
        action = np.array([*READY,1.,*READY,1.])
        trace = []

        with (output/'trace.jsonl').open('w') as file, record_trial(
                output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=RUN_LIMIT:
                    raise RuntimeError('Refill trial reached the 2,400-action limit')
                obs = env.step(SimpleNamespace(data=action)).observation
                observed = array(carton.get_transforms())[0]
                points = array(grain.get_transforms())[:,:3]
                rotation = Rotation.from_quat(observed[3:])
                relative = rotation.inv().apply(points-observed[:3])
                inside = ((np.abs(relative[:,0])<.042)&(np.abs(relative[:,1])<.027)&
                          (relative[:,2]>.003)&(relative[:,2]<.126))
                relative_station = points-origin
                h = relative_station[:,2]
                target = ((np.linalg.norm(relative_station[:,:2]-[.205,0.],axis=1)<.050)&(h>.003)&(h<.088))
                other = ((np.linalg.norm(relative_station[:,:2]-[.205,.150],axis=1)<.044)&(h>.003)&(h<.082))
                forces = array(base.scene.sensors['carton_contact'].data.normal_force_matrix_w)
                forces = np.linalg.norm(forces.reshape(-1,4,3).sum(0),axis=1)
                row = {'step':len(trace),'phase':phase,'action':action.tolist(),
                       'carton_pose_xyzw':observed.tolist(),'carton':int(inside.sum()),
                       'target':int(target.sum()),'other':int(other.sum()),
                       'outside':int((~(inside|target|other)).sum()),
                       'grain_xyz_m':points.tolist(),'fingertip_force_n':forces.tolist()}
                if lid is not None:
                    row['lid_pose_xyzw'] = array(lid.root_pose_w)[0].tolist()
                    lid_force = array(base.scene.sensors['lid_contact'].data.normal_force_matrix_w)
                    row['lid_fingertip_force_n'] = np.linalg.norm(
                        lid_force.reshape(-1,2,3).sum(0),axis=1).tolist()
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                file.write(json.dumps(row,allow_nan=False)+'\n')
                file.flush()
                frame(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%20==0:
                    Image.fromarray(obs.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,row['carton'],row['target'],
                          row['outside'],row['fingertip_force_n'],flush=True)
                if not args.dream and max(forces)>35.:
                    raise RuntimeError('Carton fingertip force exceeded 35 N')
                return obs

            if args.dream:
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)
                return run_dream(output,env,specification,env._observation,tick,audit,trace,
                                 action,RUN_LIMIT,[pantry,lid_path])

            def hold(steps, phase):
                for _ in range(steps):
                    tick(phase)

            def align(points, quats):
                start = {side:np.copy(arm_q(arm)) for side,arm in robots.items()}
                target = {}
                for side in mounts:
                    q,pe,re = ik.solve(points[side],quats[side],start[side],mounts[side],iters=400)
                    if pe>.001 or re>math.radians(.3):
                        raise RuntimeError(f'{side} overhead entry IK failed')
                    target[side] = q
                for step in range(1,61):
                    f = ik.smoothstep(step/60)
                    for i,side in enumerate(mounts):
                        action[i*7:i*7+6] = start[side]+f*(target[side]-start[side])
                    tick('align both wrists in free space')

            def move(points, quats, seconds, phase):
                working = {side:np.copy(arm_q(arm)) for side,arm in robots.items()}
                start = {side:ik.tcp_pose(q,mounts[side])[0] for side,q in working.items()}
                count = round(seconds*20)
                for step in range(1,count+1):
                    f = ik.smoothstep(step/count)
                    for i,side in enumerate(mounts):
                        point = start[side]+f*(points[side]-start[side])
                        q,pe,re = ik.solve(point,quats[side],working[side],mounts[side],iters=120)
                        if pe>.001 or re>math.radians(.3):
                            raise RuntimeError(f'{side} {phase} IK failed: {pe:.4f} m, {math.degrees(re):.2f} deg')
                        working[side] = q
                        action[i*7:i*7+6] = q
                    tick(phase)

            def relocate_lid(destination,label):
                before = np.asarray(trace[-1]['lid_pose_xyzw'])
                wrist = Rotation.from_euler('y',-20.,degrees=True)*Rotation.from_quat(DOWN_GRIP)
                pad_center = before[:3]+Rotation.from_quat(before[3:]).apply([-.065,0.,.022])
                tab = pad_center-wrist.apply([0.,0.,.010])
                clearance = np.array([0.,0.,.038])
                transit = np.array([.28,.13,.81])

                def send(joints,phase):
                    action[:6] = joints
                    tick(phase)

                def go(point,duration,phase,orient=False):
                    cartesian_move(lambda:arm_q(robots['left']),send,mounts['left'],wrist.as_quat(),
                                   point,duration,label+' '+phase,interpolate_orientation=orient)

                hover = tab+clearance
                action[6] = .55
                go(transit,2.,'align',True)
                go(hover,2.,'approach')
                go(tab,1.5,'reach handle')
                action[6] = 0.
                hold(20,label+' grasp')
                go(hover,1.6,'lift')
                raised = np.asarray(trace[-1]['lid_pose_xyzw'])
                if raised[2]-before[2]<.025 or min(trace[-1]['lid_fingertip_force_n'])<.1:
                    raise RuntimeError(label+' lid was not physically lifted')
                tcp = ik.tcp_pose(arm_q(robots['left']),mounts['left'])[0]
                place = destination+tcp-raised[:3]
                above = place+clearance
                go(transit,1.4,'cross above worktop')
                go(above,2.5,'carry')
                go(place,1.7,'seat')
                action[6] = 1.
                hold(18,label+' release')
                go(above,1.5,'retreat')
                return_start = np.copy(arm_q(robots['left']))
                for step in range(1,41):
                    f = ik.smoothstep(step/40)
                    send(return_start+f*(np.asarray(READY)-return_start),label+' park wrist')

            hold(40,'carton settling')
            reset = np.array(trace[-1]['carton_pose_xyzw'])
            if trace[-1]['carton']!=80:
                raise RuntimeError('The carton did not retain every grain after settling')
            if with_lid:
                relocate_lid(lid_park,'open jar')
            settled_rot = Rotation.from_quat(reset[3:])
            grips,quats,entry = {},{},{}
            for side,sign,roll in (('left',1.,0.),('right',-1.,180.)):
                grips[side] = reset[:3]+settled_rot.apply([-.013,sign*.057,.080])
                approach = (Rotation.from_euler('z',-90.*sign,degrees=True)*
                            Rotation.from_euler('y',90.,degrees=True)*
                            Rotation.from_euler('z',roll,degrees=True))
                quats[side] = (settled_rot*approach).as_quat()
                entry[side] = grips[side]+[0.,sign*.058,0.]
            action[[6,13]] = .65
            align({side:point+[0.,0.,.075] for side,point in entry.items()},quats)
            move(entry,quats,1.6,'descend beside grip tabs')
            move(grips,quats,1.6,'engage grip tabs')
            for width in np.linspace(.65,0.,25):
                action[[6,13]] = width
                tick('close both grippers')
            hold(12,'check both pinches')
            lifts = {side:ik.tcp_pose(arm_q(arm),mounts[side])[0]+[0.,0.,.075]
                     for side,arm in robots.items()}
            move(lifts,quats,2.,'lift with both grippers')
            hold(30,'hold carton in free space')
            held = trace[-20:]
            heights = [row['carton_pose_xyzw'][2]-reset[2] for row in held]
            forces = np.array([row['fingertip_force_n'] for row in held])
            stable = bool(min(heights)>=.065 and forces.min()>=.05 and
                          all(row['carton']==80 for row in held))
            report = {'scope':'dual gripper contact and upright lift',
                      'min_held_lift_m':float(min(heights)),
                      'min_held_fingertip_force_n':forces.min(axis=0).tolist(),
                      'min_held_grain_count':min(row['carton'] for row in held),
                      'passed':stable}
            if pour_trace and stable:
                reference = np.array(held[-1]['carton_pose_xyzw'])
                inverse = Rotation.from_quat(reference[3:]).inv()
                offsets, rotations = {}, {}
                for side,arm in robots.items():
                    position, rotation = ik.tcp_pose(arm_q(arm),mounts[side])
                    offsets[side] = inverse.apply(position-reference[:3])
                    rotations[side] = inverse*Rotation.from_matrix(rotation)

                def wrist_targets(pose):
                    rotation = Rotation.from_quat(pose[3:])
                    points = {side:pose[:3]+origin+rotation.apply(offset) for side,offset in offsets.items()}
                    quaternions = {side:(rotation*r).as_quat() for side,r in rotations.items()}
                    return points, quaternions

                target_trace = [json.loads(line) for line in pour_trace.read_text().splitlines()]
                lift_end = np.array([row['carton_pose_xyzw'] for row in target_trace if row['phase']=='lift'][-1])
                move(*wrist_targets(lift_end),1.5,'raise to pour height')
                working = {side:np.copy(arm_q(arm)) for side,arm in robots.items()}
                def follow(row,phase):
                    points, quaternions = wrist_targets(np.asarray(row['carton_pose_xyzw']))
                    for i,side in enumerate(mounts):
                        q,pe,re = ik.solve(points[side],quaternions[side],working[side],mounts[side],iters=120)
                        if pe>.001 or re>math.radians(.3):
                            raise RuntimeError(f'{side} {phase} IK failed: {pe:.4f} m, {math.degrees(re):.2f} deg')
                        working[side] = q
                        action[i*7:i*7+6] = q
                    tick(phase)

                for row in target_trace:
                    if row['phase'] in ('translate','tip_over_jar','retained_grain'):
                        follow(row,row['phase'])
                if with_lid:
                    for row in reversed(target_trace):
                        if row['phase'] in ('translate','tip_over_jar'):
                            follow(row,'return upright carton')
                    home = np.r_[reset[:3]-origin+[0.,0.,.0008],reset[3:]]
                    points, quaternions = wrist_targets(home)
                    move(points,quaternions,2.,'set carton on worktop')
                    action[[6,13]] = 1.
                    hold(20,'release carton')
                    move(entry,quats,1.5,'retreat beside carton')
                    move({side:point+[0.,0.,.075] for side,point in entry.items()},
                         quats,1.7,'raise both wrists away from carton')
                    park_start = {side:np.copy(arm_q(arm)) for side,arm in robots.items()}
                    for step in range(1,61):
                        f = ik.smoothstep(step/60)
                        for i,side in enumerate(mounts):
                            action[i*7:i*7+6] = park_start[side]+f*(np.asarray(READY)-park_start[side])
                        tick('park both wrists before closing jar')
                    relocate_lid(lid_seat,'close jar')
                    hold(25,'final settle')
                tail = trace[-20:]
                report |= {'scope':'dual gripper lift and pour into open target jar',
                           'min_target_grain_count':min(row['target'] for row in tail),
                           'max_other_grain_count':max(row['other'] for row in tail),
                           'max_outside_grain_count':max(row['outside'] for row in tail),
                           'passed':all(row['target']==80 and row['outside']==0 and row['other']==0 for row in tail)}
                if with_lid:
                    poses = np.array([row['lid_pose_xyzw'] for row in tail])
                    carton_poses = np.array([row['carton_pose_xyzw'] for row in tail])
                    seat_error = np.linalg.norm(poses[:,:3]-lid_seat,axis=1).max()
                    tilt = np.arccos(np.clip(Rotation.from_quat(poses[:,3:]).as_matrix()[:,2,2],-1.,1.)).max()
                    carton_error = np.linalg.norm(carton_poses[:,:3]-reset[:3],axis=1).max()
                    released = all(max(row['fingertip_force_n']+row['lid_fingertip_force_n'])<.1 for row in tail)
                    report |= {'scope':'open, lift, pour, return carton and reseat lid',
                               'max_lid_seat_error_m':float(seat_error),'max_lid_tilt_deg':math.degrees(tilt),
                               'max_carton_return_error_m':float(carton_error),'released':released,
                               'passed':bool(report['passed'] and seat_error<=.004 and math.degrees(tilt)<=2.
                                             and carton_error<=.018 and released)}
        if with_lid:
            report.update(audit(trace))
        return report | {'steps':len(trace)}
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pour-trace',type=Path)
    parser.add_argument('--with-lid',action='store_true')
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--dream',action='store_true')
    args = parser.parse_args()
    if args.dream:
        args.with_lid = True
    if args.with_lid and not args.dream and args.pour_trace is None:
        parser.error('--with-lid requires --pour-trace')
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
