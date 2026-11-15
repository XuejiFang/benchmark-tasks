"""Use ordinary YAM contact to turn a grinder crank around a vertical bearing."""

import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from audit_grinder_trial import CUP_HOME, INSTRUCTION, INTAKE, RUN_LIMIT, audit


def run(args):
    import isaaclab.sim as sim_utils
    import torch
    from PIL import Image
    from inspect_robots import Scene
    from yam_arena import ik, inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world

    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    origin = np.array([.31, .20, .75])
    parked = CUP_HOME
    dose_yaw = Rotation.from_euler('z',180.,degrees=True)
    intake = INTAKE

    def room(*_):
        parts = [scene._box('floor', (3.,3.,.04), (.45,0.,-.02), (.38,.4,.42)),
                scene._box('counter', (.62,.8,.04), (.49,0.,.73), (.30,.21,.16)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1300.))]
        if args.dose:
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose
            from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

            pose = Pose(position_xyz=tuple(map(float,parked)),rotation_xyzw=tuple(map(float,dose_yaw.as_quat())))
            parts.extend([
                Object(name='dose_cup',object_type=ObjectType.RIGID,
                       usd_path=str((args.dose/'cup.usda').resolve()),initial_pose=pose,
                       spawn_cfg_addon={'physics_material':scene._grippy(),'activate_contact_sensors':True,
                                        'rigid_props':PhysxRigidBodyPropertiesCfg(kinematic_enabled=False,
                                            solver_position_iteration_count=24,solver_velocity_iteration_count=8)}),
                Object(name='dose_beans',object_type=ObjectType.BASE,
                       usd_path=str((args.dose/'beans.usda').resolve()),initial_pose=pose)
            ])
        return parts

    install(args.usd,args.output,origin,(0.,0.,0.,1.), {'knob':'/Knob','base':'/Base','dial':'/Dial'},
            room_factory=room,robot_contact_offset=.001)
    if args.dose:
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        from yam_arena.robot import asset_link_paths

        original = scene.build_task_environment
        links = asset_link_paths()

        def build(*p,**kw):
            arena = original(*p,**kw)
            configure = arena.env_cfg_callback

            def config(cfg):
                cfg = configure(cfg)
                cfg.sim.dt = 1/480
                cfg.decimation = cfg.sim.render_interval = 24
                cfg.sim.use_fabric = False
                cfg.sim.physics = PhysxCfg(enable_ccd=True)
                cfg.scene.dose_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/dose_cup',
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/LeftArm'+links[part]
                                            for part in ('tip_left','tip_right')],history_length=0)
                return cfg

            arena.env_cfg_callback = config
            return arena

        scene.build_task_environment = build
    height,width = (176,320) if args.dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        device='cpu' if args.dose else 'cuda:0',
        headless=True, cam_height=height, cam_width=width, control_hz=20, object_jitter=0,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        instruction = ('Turn the grind setting collar, then turn the crank once.' if args.both else
                       'Turn the grind setting collar.' if args.dial else 'Turn the grinder crank once.')
        spec = Scene(id='grinder-crank-fixture',instruction=INSTRUCTION if args.dose else instruction)
        env.reset(spec,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        physics = None
        if args.dose:
            import omni.physics.tensors as tensors

            physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
            physics.set_subspace_roots('/')
            grains = physics.create_rigid_body_view('/World/envs/env_0/dose_beans/Bean*')
            if grains.count!=12:
                raise RuntimeError('Expected 12 free beans')
            cup = base.scene.rigid_objects['dose_cup'].data
        mechanism = MechanismObserver(base.sim.stage,physics=physics)
        robot = base.scene['left_robot']
        mount = (0.,.305,.75)
        camera = base.scene.sensors['top_cam']
        view = ((.72,-.17,1.25),(.26,.17,.84)) if args.dose else ((.07,-.26,1.16),(.29,.18,.86))
        camera.set_world_poses_from_view(torch.tensor([view[0]],device=base.device),
                                         torch.tensor([view[1]],device=base.device))
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        with (args.output/'trace.jsonl').open('w') as out, record_trial(
                args.output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=RUN_LIMIT:
                    raise RuntimeError('Grinder trial reached its action limit')
                data = env.step(SimpleNamespace(data=action)).observation
                state = mechanism.read()
                contact = {}
                for name in ('knob','base','dial'):
                    matrix = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    contact[name] = np.linalg.norm(matrix.reshape(-1,6,3).sum(0),axis=1).tolist()
                row = {'step': len(trace), 'phase': phase, 'action': action.tolist(),
                       'tcp_m': tcp_world(robot).tolist(), 'contacts_n': contact, **state}
                if args.dose:
                    points = array(grains.get_transforms())[:,:3]
                    relative = points-intake
                    inside = ((np.linalg.norm(relative[:,:2],axis=1)<.034)&
                              (relative[:,2]>.004)&(relative[:,2]<.090))
                    matrix = array(base.scene.sensors['dose_contact'].data.normal_force_matrix_w)
                    row.update(bean_xyz_m=points.tolist(),beans_in_intake=int(inside.sum()),
                               cup_pose_xyzw=array(cup.root_pose_w)[0].tolist(),
                               cup_contact_n=np.linalg.norm(matrix.reshape(-1,2,3).sum(0),axis=1).tolist())
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                frame(data.images['top_cam'])
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                trace.append(row)
                if len(trace) == 1 or len(trace)%30 == 0:
                    Image.fromarray(data.images['top_cam']).save(args.output/'latest.png')
                    print('STEP',len(trace),phase,flush=True)
                if not args.dream and (sum(contact['base'])>20. or sum(contact['knob'])>110. or sum(contact['dial'])>50.):
                    raise RuntimeError('Grinder contact guard exceeded')
                return data

            if args.dream:
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)
                assets = [args.usd,args.dose/'cup.usda',args.dose/'beans.usda']
                return run_dream(args.output,env,spec,env._observation,tick,audit,trace,
                                 action,RUN_LIMIT,assets)
            def send(q, phase):
                action[:6] = q
                tick(phase)

            def move(point, seconds, phase, orient=False):
                cartesian_move(lambda:arm_q(robot),send,mount,DOWN_GRIP,point,
                               seconds,phase,interpolate_orientation=orient)

            def deliver_dose():
                reset = np.array(trace[-1]['cup_pose_xyzw'])
                handle = reset[:3]+Rotation.from_quat(reset[3:]).apply([-.034,0.,.027])
                action[6] = .55
                move(handle+[0.,0.,.055],2.6,'approach dose handle',True)
                move(handle,1.4,'enter dose grip')
                action[6] = 0.
                for _ in range(20):
                    tick('grasp dose cup')
                move(handle+[0.,0.,.073],2.,'lift dose cup')
                held = np.array(trace[-1]['cup_pose_xyzw'])
                if held[2]-reset[2]<.055 or min(trace[-1]['cup_contact_n'])<.1:
                    raise RuntimeError('Dose cup failed to lift with two fingertips')
                tcp, matrix = ik.tcp_pose(arm_q(robot),mount)
                r = Rotation.from_quat(held[3:])
                offset = r.inv().apply(tcp-held[:3])
                wrist = r.inv()*Rotation.from_matrix(matrix)
                lip = np.array([.048,0.,.011])

                def command(position,rotation,seconds,phase):
                    cartesian_move(lambda:arm_q(robot),send,mount,(rotation*wrist).as_quat(),
                                   position+rotation.apply(offset),seconds,phase,
                                   interpolate_orientation=True)

                mouth = intake+dose_yaw.apply([-.015,0.,.145])
                level = mouth-dose_yaw.apply(lip)
                command(reset[:3]+[0.,0.,.125],dose_yaw,1.5,'bring dose cup above the rim')
                command(level,dose_yaw,2.4,'carry dose to intake')

                def tilt(fractions,phase,stop_when_full=False):
                    q = np.copy(arm_q(robot))
                    for t in fractions:
                        f = ik.smoothstep(t)
                        rotation = dose_yaw*Rotation.from_euler('y',94.*f,degrees=True)
                        position = mouth-rotation.apply(lip)+[0.,0.,-.026*f]
                        q,pe,re = ik.solve(position+rotation.apply(offset),
                                           (rotation*wrist).as_quat(),q,mount,iters=120)
                        if pe>.001 or re>math.radians(.3):
                            raise RuntimeError(f'{phase}: IK failed at {94.*f:.1f} degrees')
                        send(q,phase)
                        if stop_when_full and all(row['beans_in_intake']==12 for row in trace[-5:]):
                            break
                    return t

                peak = tilt(np.linspace(0.,1.,81)[1:],'pour beans into intake',stop_when_full=True)
                for _ in range(35):
                    tick('allow beans to settle')
                tilt(np.linspace(peak,0.,41)[1:],'level empty cup')
                command(reset[:3]+[0.,0.,.124],dose_yaw,2.4,'return dose cup above the rim')
                command(reset[:3]+[0.,0.,.001],Rotation.from_quat(reset[3:]),1.7,'park empty cup')
                action[6] = 1.
                for _ in range(18):
                    tick('release dose cup')
                move(tcp_world(robot)+[0.,0.,.045],1.4,'clear dose cup',True)

            for _ in range(30):
                tick('settle')
            initial = trace[-1]
            if args.dial or args.both:
                pivot = np.array(initial['bodies']['/Dial']['pose_xyzw'][:3])
                x = -args.dial_contact_radius
                depth = .011 if args.dial_contact_radius>.04 else .014
                contact_y = .008 if args.dial_contact_radius>.04 else .015
                action[6] = 0.
                move(pivot+[x,.033,.033],3.,'approach dial from above',True)
                move(pivot+[x,.033,depth],1.5,'lower beside dial pointer')
                move(pivot+[x,contact_y+.004,depth],1.0,'meet dial pointer')
                stop_angle = math.radians(29.4 if args.dose else 55.)
                for _ in range(160):
                    joints = next(iter(trace[-1]['articulations'].values()))
                    actual = joints['positions'][joints['names'].index('GrindSize')]
                    if actual >= stop_angle:
                        break
                    turn = Rotation.from_euler('z',actual+math.radians(3.))
                    point = pivot+turn.apply([x,contact_y,depth])
                    q,pe,re = ik.solve(point,DOWN_GRIP,arm_q(robot),base_pos=mount,iters=100)
                    if pe>.001 or re>math.radians(.3):
                        raise RuntimeError('Dial sweep IK failed')
                    send(q,'push dial pointer')
                move(tcp_world(robot)+[0.,0.,.046],1.4,'clear dial pointer')
                action[6] = 1.
                for _ in range(20):
                    tick('settle after dial adjustment')
            if not args.dial:
                if args.dose:
                    deliver_dose()
                after_dial = trace[-1]
                pose = np.array(after_dial['bodies']['/Crank']['pose_xyzw'])
                grasp = np.array(after_dial['bodies']['/Knob']['pose_xyzw'][:3])+[0.,0.,.007]
                move(grasp+[0.,0.,.033 if args.dose else .040],3.,'approach crank',True)
                move(grasp,1.5,'lower to knob')
                action[6] = 0.
                for _ in range(25):
                    tick('close on crank knob')
                if min(trace[-1]['contacts_n']['knob'][1:3]) < .1:
                    raise RuntimeError('No bilateral contact on crank knob')
                radial = tcp_world(robot)-pose[:3]
                joints = next(iter(trace[-1]['articulations'].values()))
                crank_index = joints['names'].index('CrankRotation')
                previous = joints['positions'][crank_index]
                progress = 0.
                total_turn = math.radians(380.) if args.dose else math.tau
                for index in range(1,451):
                    planned = total_turn*ik.smoothstep(min(index/320,1.))
                    turn = Rotation.from_euler('z',min(planned,progress+math.radians(12.)))
                    point = pose[:3]+turn.apply(radial)
                    q,pe,re = ik.solve(point, DOWN_GRIP,arm_q(robot),base_pos=mount,iters=100)
                    if pe > .001 or re > math.radians(.3):
                        raise RuntimeError('Crank sweep IK failed')
                    send(q,'turn crank')
                    joints = next(iter(trace[-1]['articulations'].values()))
                    measured = joints['positions'][crank_index]
                    progress += math.remainder(measured-previous, math.tau)
                    previous = measured
                    if progress >= math.radians(358.):
                        break
                action[6] = 1.
                for _ in range(20):
                    tick('release knob')
                clear = tcp_world(robot)+[0.,0.,.030 if args.dose else .035]
                move(clear,1.5,'lift open gripper above knob')
                move(clear+[0.,.065,0.],1.5,'move clear of grinder')
            for _ in range(25):
                tick('final settle')
        joint_series = [next(iter(row['articulations'].values())) for row in trace]
        names = joint_series[0]['names']
        joint_motion = np.rad2deg(np.unwrap([j['positions'] for j in joint_series],axis=0))
        dial_history = joint_motion[:,names.index('GrindSize')]
        crank_history = joint_motion[:,names.index('CrankRotation')]
        transfer = next((i for i,row in enumerate(trace) if row['phase']=='approach crank'),len(trace)-1)
        report = {'scope':('fixed-base dial and crank contact' if args.both else
                         'fixed-base dial contact' if args.dial else 'fixed-base crank contact'),
                'steps':len(trace),
                'joint_displacement_deg':dict(zip(names,(joint_motion[-1]-joint_motion[29]).tolist())),
                'dial_motion_during_crank_deg':float(np.max(abs(dial_history[transfer:]-dial_history[transfer]))),
                'crank_motion_after_dial_deg':float(crank_history[-1]-crank_history[transfer]),
                'max_final_gripper_contact_n':max(sum(r['contacts_n'][part]) for r in trace[-20:] for part in ('dial','knob'))}
        if args.dose:
            report['min_final_beans_in_intake'] = min(r['beans_in_intake'] for r in trace[-20:])
            report.update(audit(trace))
        return report
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dial', action='store_true')
    modes.add_argument('--both', action='store_true')
    parser.add_argument('--dose',type=Path)
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--dial-contact-radius',type=float,default=.038)
    args = parser.parse_args()
    if args.dose and not args.both:
        parser.error('--dose requires --both')
    if args.dream and not args.dose:
        parser.error('--dream requires --dose')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True).app
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
