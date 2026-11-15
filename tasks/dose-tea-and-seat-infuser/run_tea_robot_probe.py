"""Run the dry-tea scoop as a free rigid tool held by the YAM fingertips."""

import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


INSTRUCTION = (
    'Take the lid off the green tea tin. Set it flat on the worktop about ten centimetres '
    'diagonally to the right and away from the robot bases. Use the scoop to put 24 to 33 '
    'pieces of dry tea from the tin into the loose metal filter basket. Do not spill any tea. '
    'Set down the scoop, then seat the filled filter basket upright in the brown cup and release it.'
)


def run(args):
    import torch
    from PIL import Image
    from inspect_robots import Scene
    from isaaclab.sensors import ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    import omni.physics.tensors as tensors
    from yam_arena import embodiment, ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation
    from yam_arena.robot import asset_link_paths
    from yam_arena.runtime import arm_q

    from articulated_fixture import array
    from expert_motion import cartesian_move
    from record_trial import record_trial

    origin = np.array([.27,.20,.75])
    mount = (0.,.305,.83)
    station.BASES = {'left':mount, 'right':(0.,-.305,.83)}
    alignment = Rotation.from_euler('z',90,degrees=True)
    down = (alignment*Rotation.from_matrix(grip_rotation(0))).as_quat()
    parked = np.array([.22,.26,.751])
    pose = tuple(map(float,alignment.as_quat()))
    filter_start = origin+alignment.apply([.145,-.017,.001])
    cup = origin+alignment.apply([.12,.09,0.])

    def rigid(name, path, position):
        return Object(name=name,object_type=ObjectType.RIGID,usd_path=str(path.resolve()),
                      initial_pose=Pose(position_xyz=tuple(map(float,position)),rotation_xyzw=pose),
                      spawn_cfg_addon={'physics_material':scene._grippy(),'activate_contact_sensors':True,
                                       'rigid_props':PhysxRigidBodyPropertiesCfg(
                                           kinematic_enabled=False,solver_position_iteration_count=24,
                                           solver_velocity_iteration_count=8,max_depenetration_velocity=.3)})

    def room(*_):
        parts = [scene._box('worktop',(.60,.75,.04),(.38,.13,.73),(.36,.24,.16)),
                 scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.34,.37,.39)),
                 scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1300.))]
        parts.extend([Object(name='tea',object_type=ObjectType.BASE,usd_path=str(args.tea.resolve()),
                             initial_pose=Pose(position_xyz=tuple(map(float,origin)),rotation_xyzw=pose)),
                      rigid('scoop',args.scoop,parked)])
        if args.accessories:
            parts.extend([rigid('cap',args.accessories/'cap.usda',origin+[0.,0.,.0558]),
                          rigid('filter',args.accessories/'filter.usda',filter_start)])
        return parts

    scene.task_room = room
    original = scene.build_task_environment
    links = asset_link_paths()

    def build(*positional, **keywords):
        arena = original(*positional,**keywords)
        configure = arena.env_cfg_callback

        def config(cfg):
            cfg = configure(cfg)
            cfg.sim.dt = 1/480
            cfg.decimation = cfg.sim.render_interval = 24
            cfg.sim.use_fabric = False
            cfg.sim.physics = PhysxCfg(enable_ccd=True)
            cfg.scene.scoop_contact = ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/scoop',
                filter_prim_paths_expr=['{ENV_REGEX_NS}/LeftArm'+links[part] for part in ('tip_left','tip_right')],
                history_length=0)
            return cfg

        arena.env_cfg_callback = config
        return arena

    scene.build_task_environment = build
    original_embodiment = embodiment.BimanualYamEmbodiment
    embodiment.BimanualYamEmbodiment = lambda *a,**kw: original_embodiment(*a,**(kw | {'robot_contact_offset':.001}))
    env = inspect_embodiment.YamArenaEmbodiment(
        device='cpu',headless=True,cam_height=176 if args.dream else 360,
        cam_width=320 if args.dream else 640,control_hz=20,object_jitter=0,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        scene_spec = Scene(id='prepare-dry-tea-dose',instruction=INSTRUCTION)
        env.reset(scene_spec,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        root = base.scene.rigid_objects['scoop'].data
        parts = {name:base.scene.rigid_objects[name].data for name in ('cap','filter')} if args.accessories else {}
        robot = base.scene['left_robot']
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.50,-.13,1.16]],device=base.device),
                                         torch.tensor([[.25,.26,.79]],device=base.device))
        sim = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
        sim.set_subspace_roots('/')
        leaves = sim.create_rigid_body_view('/World/envs/env_0/tea/Leaf*')
        if leaves.count != 160:
            raise RuntimeError(f'Expected 160 separate dry leaves, found {leaves.count}')
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        grip_local = np.array([-.013,0.,.053])
        with (args.output/'trace.jsonl').open('w') as out, record_trial(
                args.output,True,width=env.cam_width,height=env.cam_height) as frame:
            def tick(phase, inference_latency=None):
                if len(trace)>=2400:
                    raise RuntimeError('Tea trial reached the 2,400-action limit')
                obs = env.step(SimpleNamespace(data=action)).observation
                tool = array(root.root_pose_w)[0]
                xyz = alignment.inv().apply(array(leaves.get_transforms())[:,:3]-origin)
                assets = {name:array(data.root_pose_w)[0] for name,data in parts.items()}
                if assets:
                    filter_pose = assets['filter']
                    frame_xyz = Rotation.from_quat(filter_pose[3:]).inv().apply(
                        alignment.apply(xyz)+origin-filter_pose[:3])
                else:
                    frame_xyz = xyz-[.145,-.017,0.]
                forces = array(base.scene.sensors['scoop_contact'].data.normal_force_matrix_w)
                forces = np.linalg.norm(forces.reshape(-1,2,3).sum(0),axis=1)
                tin = (np.linalg.norm(xyz[:,:2],axis=1)<.037)&(xyz[:,2]>.004)&(xyz[:,2]<.055)
                basket = (np.linalg.norm(frame_xyz[:,:2],axis=1)<.028)&(frame_xyz[:,2]>.004)&(frame_xyz[:,2]<.033)
                row = {'step':len(trace),'phase':phase,'action':action.tolist(),'tool_pose_xyzw':tool.tolist(),
                       'gripper_contact_n':forces.tolist(),'leaf_positions_station_m':xyz.tolist(),
                       'tin':int(tin.sum()),'filter':int(basket.sum()),
                       'parts':{name:value.tolist() for name,value in assets.items()}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                frame(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%20==0:
                    Image.fromarray(obs.images['top_cam']).save(args.output/'latest.png')
                    print('STEP',len(trace),phase,row['tin'],row['filter'],flush=True)
                return obs

            if args.dream:
                from audit_tea_robot_trial import audit
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)
                assets = (args.tea,args.scoop,args.accessories/'cap.usda',args.accessories/'filter.usda')
                return run_dream(args.output,env,scene_spec,env._observation,tick,audit,trace,action,2400,assets)

            def send(q, phase):
                action[:6] = q
                tick(phase)

            def move(point, seconds, phase, interpolate=False, quat=down):
                cartesian_move(lambda:arm_q(robot),send,mount,quat,point,seconds,phase,
                               interpolate_orientation=interpolate)

            def relocate(name, grip_offset, destination, height):
                before = np.array(trace[-1]['parts'][name])
                handle = before[:3]+Rotation.from_quat(before[3:]).apply(grip_offset)
                move(handle+[0.,0.,.028],2.5,'approach '+name,True)
                action[6] = .45
                move(handle,1.4,'enter '+name+' grasp')
                action[6] = 0.
                for _ in range(16):
                    tick('grasp '+name)
                move(handle+[0.,0.,height],1.7,'lift '+name)
                held = np.array(trace[-1]['parts'][name])
                if held[2]-before[2] < .025:
                    raise RuntimeError(f'{name} did not lift with the fingers')
                tcp,gripper = ik.tcp_pose(arm_q(robot),mount)
                asset_to_gripper = Rotation.from_quat(held[3:]).inv()*Rotation.from_matrix(gripper)
                target_rotation = alignment*asset_to_gripper
                center_offset = Rotation.from_matrix(gripper).inv().apply(held[:3]-tcp)
                put = np.asarray(destination)-target_rotation.apply(center_offset)
                hover = put.copy()
                hover[2] = max(tcp[2],put[2]+.046)
                quat = target_rotation.as_quat()
                move(hover,2.8,'carry '+name,True,quat)
                measured = np.array(trace[-1]['parts'][name][:3])
                current,_ = ik.tcp_pose(arm_q(robot),mount)
                correction = np.asarray(destination)-measured
                level = current.copy()
                level[:2] += correction[:2]
                move(level,1.2,'center '+name,quat=quat)
                measured = np.array(trace[-1]['parts'][name][:3])
                current,_ = ik.tcp_pose(arm_q(robot),mount)
                put = current+np.asarray(destination)-measured
                if name == 'filter':
                    q = arm_q(robot)
                    previous = measured
                    offset = current-measured
                    for i in range(1,91):
                        actual = np.array(trace[-1]['parts']['filter'][:3])
                        fraction = ik.smoothstep(i/90)
                        target = measured+fraction*(np.asarray(destination)-measured)
                        correction = np.clip(.65*(previous-actual),-.003,.003)
                        previous = target
                        q,pe,re = ik.solve(target+offset+correction,quat,q,base_pos=mount,iters=100)
                        if pe>.001 or re>math.radians(.3):
                            raise RuntimeError('Filter insertion exceeded wrist IK tolerance')
                        send(q,'lower filter')
                    put,_ = ik.tcp_pose(arm_q(robot),mount)
                else:
                    move(put,2.,'lower '+name,quat=quat)
                action[6] = 1.
                for _ in range(20):
                    tick('release '+name)
                move(put+[0.,0.,.035],1.4,'clear '+name,quat=quat)

            for _ in range(40):
                tick('settle')
            if args.accessories:
                relocate('cap',[0.,0.,.020],[.34,.12,.753],.047)
            initial = np.array(trace[-1]['tool_pose_xyzw'])
            grip = initial[:3]+Rotation.from_quat(initial[3:]).apply(grip_local)
            move(grip+[0.,0.,.032],2.5,'approach scoop',True)
            action[6] = .4
            move(grip,1.5,'align fingers with handle')
            action[6] = 0.
            for _ in range(20):
                tick('grasp scoop handle')
            if min(trace[-1]['gripper_contact_n']) < .12:
                raise RuntimeError('Expected contact on both sides of the scoop handle')
            move(grip+[0.,0.,.061],2.,'lift scoop')
            held = np.array(trace[-1]['tool_pose_xyzw'])
            if held[2] < initial[2]+.05:
                raise RuntimeError('The scoop did not lift with the gripper')
            p,r = ik.tcp_pose(arm_q(robot),mount)
            wrist = Rotation.from_matrix(r)
            local_pose = Rotation.from_quat(held[3:])
            handle_at = local_pose.inv().apply(p-held[:3])
            relative_rotation = local_pose.inv()*wrist
            bias = np.zeros(3)

            def follow(start, end, steps, phase):
                q = arm_q(robot)
                previous = origin+alignment.apply(start[:3])
                for i in range(1,steps+1):
                    x,y,z,deg = start+ik.smoothstep(i/steps)*(end-start)
                    tool_rotation = alignment*Rotation.from_euler('y',deg,degrees=True)
                    # Refer to the observed scoop, so tracking and contact compliance
                    # cannot leave the shallow lip above the bottom layer of leaves.
                    observed = np.array(trace[-1]['tool_pose_xyzw'][:3])
                    bias[:] = np.clip(.86*bias+.24*(previous-observed),-.006,.006)
                    previous = origin+alignment.apply([x,y,z])
                    target = previous+tool_rotation.apply(handle_at)+bias
                    q, pe, re = ik.solve(target,(tool_rotation*relative_rotation).as_quat(),q,base_pos=mount,iters=100)
                    if pe>.001 or re>math.radians(.3):
                        raise RuntimeError(f'{phase} IK error {pe:.5f} m, {math.degrees(re):.3f} deg')
                    send(q,phase)

            start = np.r_[alignment.inv().apply(held[:3]-origin),0.]
            def execute(path):
                nonlocal start
                for point, steps, phase in path:
                    destination = np.array(point)
                    follow(start,destination,steps,phase)
                    start = destination

            def scoop_and_pour(dose_x):
                execute([([-.019,0.,.080,0.],35,'carry over tea'),([-.019,0.,.0055,0.],55,'enter tea tin'),
                         ([dose_x,0.,.0055,0.],60,'scoop leaves'),([dose_x,0.,.055,0.],48,'lift in tin'),
                         ([dose_x,0.,.055,-9.],38,'tilt back in tin'),([dose_x,0.,.083,-9.],40,'lift dose'),
                         ([dose_x,0.,.083,-9.],25,'settle scoop above tin')])
                if args.trim_dose and 160-trace[-1]['tin']>33:
                    # Keep the lip over the open tin and return surplus by gravity.
                    # Observe the settled inventory between small rotations.
                    for angle in (4.,9.,13.,17.,21.,25.,29.,33.):
                        execute([([dose_x,0.,.083,angle],12,'return excess to tin')])
                        if 160-trace[-1]['tin']<=32:
                            break
                    execute([([dose_x,0.,.083,-9.],27,'level retained dose')])
                execute([([.132,-.017,.068,-9.],130,'carry dose'),([.132,-.017,.055,65.],56,'pour dose'),
                         ([.132,-.017,.055,65.],50,'settle deposited leaves')])

            scoop_and_pour(args.scoop_end_x)
            deposited = trace[-1]
            if args.top_up and (deposited['filter']+deposited['tin']==160 and deposited['filter']<24):
                execute([([.131,-.017,.095,0.],52,'clear filter for another scoop')])
                scoop_and_pour(.0105 if deposited['filter']>=16 else args.scoop_end_x)
            if args.accessories:
                execute([([.131,-.017,.095,0.],52,'clear poured dose'),
                         ([.060,.050,.060,0.],55,'carry empty scoop back'),
                         ([.060,.050,.001,0.],50,'lower scoop to worktop')])
                action[6] = 1.
                for _ in range(24):
                    tick('release empty scoop')
                tcp,_ = ik.tcp_pose(arm_q(robot),mount)
                move(tcp+[0.,0.,.050],1.5,'clear parked scoop',True)
                relocate('filter',[-.039,0.,.046],cup+[0.,0.,.0175],.063)
                for _ in range(32):
                    tick('settle filter in cup')
        return {'scope':'free scoop manipulated through YAM gripper contact','steps':len(trace),
                'filter_min_last_second':min(row['filter'] for row in trace[-20:]),
                'tin_min_last_second':min(row['tin'] for row in trace[-20:]),
                'accounted_at_finish':trace[-1]['tin']+trace[-1]['filter'],
                'final_accessory_pose_xyzw':trace[-1]['parts'],'cup_position_m':cup.tolist()}
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tea',type=Path,required=True)
    parser.add_argument('--scoop',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--accessories',type=Path)
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--top-up',action='store_true',help='Try a second dose if the first is short and contained.')
    parser.add_argument('--trim-dose',action='store_true',help='Return measured surplus to the tin by tilting the scoop.')
    parser.add_argument('--scoop-end-x',type=float,default=.0165)
    args = parser.parse_args()
    if not .008<=args.scoop_end_x<=.017:
        parser.error('--scoop-end-x exceeds the verified tin clearance envelope')
    if args.dream and args.accessories is None:
        parser.error('Dream requires the full scene with a cap and a movable filter basket')
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
