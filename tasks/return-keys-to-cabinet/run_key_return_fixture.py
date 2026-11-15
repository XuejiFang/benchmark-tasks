"""Measure stable tray resets and open a real hinge via the YAM gripper."""
import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


INSTRUCTION = (
    'Open the wall-mounted key cabinet. Hang the blue key ring on the left hook '
    'and the orange key ring on the right hook as seen from the front. Seat each '
    'key blade in the guide directly under its hook. Release both keys and close the door.'
)
RUN_LIMIT = 2400


def run(args):
    import isaaclab.sim as sim_utils
    import torch
    from PIL import Image
    from inspect_robots import Scene
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    from yam_arena import ik, inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation
    from yam_arena.robot import asset_link_paths
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from key_return_score import HOOK_Y, settled_hangs
    from record_trial import record_trial

    both = args.phase == 'return-both'
    cabinet_y = -.06 if both else 0.
    origin = np.array([args.cabinet_x, cabinet_y, .76])
    # Keep the tray rim below the cabinet door's 0.768 m bottom edge,
    # with clearance for both colliders' contact envelopes.
    tray_top = .723
    grip_pose = Rotation.from_euler('y', 90, degrees=True) * Rotation.from_euler('z', -90, degrees=True)
    names = ('blue_key', 'orange_key')
    # Keep the complete 102 mm shaft inside the tray. The older 170 mm offset
    # lodged the blade across the rim and arrested the actual wrist during lift.
    blue_start = ((.365,-.05,tray_top+.009),180.) if both else ((.305,.117,tray_top+.009),0.)
    top_levels = [blue_start, ((.305, -.117, tray_top + .009), 180.)]
    asset_poses = dict(zip(names, top_levels))

    def key(spec):
        position, yaw = asset_poses[spec.name]
        # Local key plane XZ is turned parallel to the tray with the fob underside down.
        lay_flat = Rotation.from_euler('x', 90, degrees=True)
        orientation = Rotation.from_euler('z', yaw, degrees=True) * lay_flat
        if args.phase == 'support-control':
            hook_y = HOOK_Y[spec.name]
            # A second freely falling key is a near-miss control, displaced
            # sideways far enough that no shaft crosses its aperture.
            if spec.name == 'orange_key':
                hook_y -= .044
            position = (args.cabinet_x+.112, hook_y, 1.004)
            orientation = Rotation.from_euler('z', 90, degrees=True)
        return Object(
            name=spec.name, object_type=ObjectType.RIGID,
            usd_path=str((args.assets / f'{spec.name}.usda').resolve()),
            initial_pose=Pose(position_xyz=position, rotation_xyzw=tuple(map(float, orientation.as_quat()))),
            spawn_cfg_addon={
                'physics_material': scene._grippy(),
                'activate_contact_sensors': True,
                'collision_props': sim_utils.CollisionPropertiesCfg(contact_offset=args.key_offset, rest_offset=0.),
                'rigid_props': PhysxRigidBodyPropertiesCfg(
                    solver_position_iteration_count=32 if args.fine_contact else 16,
                    solver_velocity_iteration_count=12 if args.fine_contact else 4,
                    max_depenetration_velocity=.08 if args.fine_contact else .25, linear_damping=.06,
                    angular_damping=4. if args.fine_contact else (.9 if both else .25),
                ),
            },
        )

    def room(*_):
        box = scene._box
        supports = [
            box('floor', (3., 3., .04), (.45, 0., -.02), (.32, .34, .35)),
            box('wall', (.09, 1.2, 1.65), (args.cabinet_x + .214, 0., .825), (.77, .78, .73)),
            box('console', (.32, .56, .05), (.31, 0., .686), (.35, .23, .13)),
            box('tray_floor', (.26, .48, .012), (.31, 0., tray_top - .006), (.21, .22, .23)),
        ]
        for side, y in [('left', .305), ('right', -.305)]:
            supports.append(box(side+'_pedestal', (.18, .18, .75), (0., y, .375), (.22, .24, .27)))
        for side, y in [('left', .235), ('right', -.235)]:
            supports.append(box('tray_'+side, (.26, .01, .018), (.31, y, tray_top+.009), (.21,.22,.23)))
        for edge, x in [('front', .185), ('back', .435)]:
            supports.append(box('tray_'+edge, (.01, .46, .018), (x, 0., tray_top+.009), (.21,.22,.23)))
        return [*supports, scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1100))]

    scene.task_object = key
    install(args.assets/'key_cabinet.usda', args.output, origin, (0., 0., 0., 1.),
            {'case': '/Case', 'door': '/Door'}, room_factory=room, robot_contact_offset=.001)
    previous_builder = scene.build_task_environment
    finger_links = asset_link_paths()

    def instrument(*positional, **named):
        arena = previous_builder(*positional,**named)
        configure = arena.env_cfg_callback

        def add_key_contacts(cfg):
            cfg = configure(cfg)
            if args.fine_contact:
                cfg.sim.dt = 1/480
                cfg.decimation = cfg.sim.render_interval = 24
            for name in names:
                setattr(cfg.scene,'key_contact_'+name,ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/'+name,
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/'+side+finger_links[part]
                                            for side in ('LeftArm','RightArm')
                                            for part in ('gripper','tip_left','tip_right')],
                    history_length=0))
            return cfg

        arena.env_cfg_callback = add_key_contacts
        return arena

    scene.build_task_environment = instrument
    height, width = (176,320) if args.dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True, cam_height=height, cam_width=width, control_hz=20,
        task=str(Path(__file__).with_name('key_return_fixture_task.py')), object_jitter=0,
    )
    try:
        specification = Scene(id='key-return-fixture',instruction=INSTRUCTION if both else
                              'Open the entryway key cabinet.')
        env.reset(specification,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        mechanism = MechanismObserver(base.sim.stage)
        robot = base.scene['right_robot']
        mount = (0., -.305, .75)
        action = np.array([*READY, 1., *READY, 1.])
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(
            torch.tensor([[-.70, -1.06, 1.37]], device=base.device),
            torch.tensor([[.43, 0., .91]], device=base.device))
        trace = []

        with (args.output/'trace.jsonl').open('w') as out, record_trial(
                args.output,args.video or args.dream,width=width,height=height) as write_frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=RUN_LIMIT:
                    raise RuntimeError('Key return trial reached the action limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                if write_frame is not None:
                    write_frame(observation.images['top_cam'])
                state = mechanism.read()
                if len(state['articulations']) != 1:
                    raise RuntimeError('Expected one live cabinet articulation')
                hinge = next(iter(state['articulations'].values()))['positions'][0]
                keys = {}
                for name in names:
                    data = base.scene.rigid_objects[name].data
                    sensor = base.scene.sensors['key_contact_'+name].data.normal_force_matrix_w
                    force = np.linalg.norm(array(sensor).reshape(-1,6,3).sum(0),axis=1).tolist()
                    keys[name] = {'pose_xyzw': array(data.root_pose_w)[0].tolist(),
                                  'velocity': array(data.root_vel_w)[0].tolist(),
                                  'gripper_force_n': force}
                contacts = {}
                for name in ('case', 'door'):
                    sensor = base.scene.sensors['mechanism_contact_'+name].data
                    contacts[name] = np.linalg.norm(
                        array(sensor.normal_force_matrix_w).reshape(-1,6,3).sum(0), axis=1).tolist()
                row = {'step': len(trace), 'phase': phase, 'action': action.tolist(),
                       'hinge_deg': math.degrees(hinge), 'keys': keys,
                       'tcp_m': tcp_world(robot).tolist(),
                       'left_tcp_m': tcp_world(base.scene['left_robot']).tolist(),
                       'contacts_n': contacts,
                       'door_pose_xyzw': state['bodies']['/Door']['pose_xyzw']}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                out.write(json.dumps(row, allow_nan=False)+'\n')
                out.flush()
                trace.append(row)
                if len(trace) in (1, 80) or len(trace)%30 == 0:
                    frame = Image.fromarray(np.asarray(observation.images['top_cam']))
                    frame.save(args.output/'latest.png')
                    if phase == 'swing':
                        frame.save(args.output/f'swing_{len(trace):03}.png')
                    print('STEP', len(trace), phase, round(row['hinge_deg'], 2), flush=True)
                if not args.dream and (sum(contacts['case']) > 20 or sum(contacts['door']) > 100):
                    raise RuntimeError('Cabinet contact guard exceeded')
                return observation

            if args.dream:
                from audit_key_return_trial import audit
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)

                def score(rows):
                    return audit(rows,args.cabinet_x,cabinet_y)

                assets = [args.assets/file for file in ('key_cabinet.usda','blue_key.usda','orange_key.usda')]
                return run_dream(args.output,env,specification,env._observation,tick,score,
                                 trace,action,RUN_LIMIT,assets)

            for _ in range(160 if args.phase == 'support-control' else 80):
                tick('tray settle')
            still = {
                name: bool(
                    max(np.linalg.norm(row['keys'][name]['velocity'][:3]) for row in trace[-20:]) < .005
                    and max(np.linalg.norm(row['keys'][name]['velocity'][3:]) for row in trace[-20:]) < .05
                )
                for name in names
            }
            return_phase = args.phase in ('return-blue', 'return-orange', 'return-both')
            if args.phase in ('door','close-probe') or return_phase:
                pose = np.array(trace[-1]['door_pose_xyzw'])
                hinge_point = pose[:3]
                local_handle = np.array([-.058, -.289, 0.])
                bar = Rotation.from_quat(pose[3:]).apply(local_handle)
                grasp = hinge_point + bar + [-.004, 0., 0.]

                def send(q, phase):
                    action[7:13] = q
                    tick(phase)

                def move(point, seconds, phase, orient=False):
                    cartesian_move(lambda: arm_q(robot), send, mount, grip_pose.as_quat(),
                                   point, seconds, phase, interpolate_orientation=orient)

                def swing_door(degrees, phase, stop_at=None):
                    # The round handle can roll relative to the fingertips as the door turns.
                    radial = tcp_world(robot) - hinge_point
                    for i in range(1, 111):
                        turn = Rotation.from_euler('z', degrees*ik.smoothstep(i/110), degrees=True)
                        point = hinge_point + turn.apply(radial)
                        q, pos_error, angle_error = ik.solve(point, grip_pose.as_quat(), arm_q(robot),
                                                             base_pos=mount, iters=100)
                        if pos_error > .001 or angle_error > math.radians(.3):
                            raise RuntimeError(f'{phase} IK residual {pos_error:.5f} m')
                        send(q, phase)
                        if stop_at is not None and trace[-1]['hinge_deg'] >= stop_at:
                            return

                move(grasp + [-.065, 0., 0.], 3.0, 'handle approach', True)
                move(grasp, 1.5, 'enter handle')
                action[13] = 0.
                for _ in range(25):
                    tick('close gripper')
                if min(trace[-1]['contacts_n']['door'][4:6]) < .1:
                    raise RuntimeError('No bilateral finger contact on the handle')
                swing_door(-args.open_degrees, 'swing')
                action[13] = 1.
                for _ in range(25):
                    tick('release')
                move(tcp_world(robot) + [-.035, .02, 0.], 1., 'withdraw')
                for _ in range(40):
                    tick('open-door settle')
                if return_phase or args.phase == 'close-probe':
                    move([.22, -.235, .93], 3., 'park right wrist away from opening')

            def push_door_shut():
                left = base.scene['left_robot']
                left_mount = (0.,.305,.75)
                forward = Rotation.from_euler('y',90,degrees=True)

                def send_left(q, name):
                    action[:6] = q
                    tick(name)

                def panel_goal(angle, depth):
                    panel = Rotation.from_euler('z',angle,degrees=True)
                    reach = .19 if both else .22
                    return hinge_point+panel.apply([depth,-reach,.015]), (panel*forward).as_quat()

                angle = trace[-1]['hinge_deg']
                point, quat = panel_goal(angle,-.068)
                cartesian_move(lambda:arm_q(left),send_left,left_mount,quat,point,3.,
                               'approach cabinet outer face',interpolate_orientation=True)
                action[6] = 0.
                for _ in range(12):
                    tick('close fingers for panel push')
                point, quat = panel_goal(trace[-1]['hinge_deg'],-.031)
                cartesian_move(lambda:arm_q(left),send_left,left_mount,quat,point,1.8,'meet outer panel')
                for _ in range(230):
                    measured = trace[-1]['hinge_deg']
                    if measured >= -1.5:
                        break
                    predicted = min(0.,measured+1.2)
                    point, quat = panel_goal(predicted,-.027)
                    q,pe,re = ik.solve(point,quat,arm_q(left),base_pos=left_mount,iters=100)
                    if pe > .001 or re > math.radians(.3):
                        raise RuntimeError(f'Door panel approach exceeded wrist IK tolerance at {measured:.2f} deg')
                    send_left(q,'push outer panel')
                point, quat = panel_goal(trace[-1]['hinge_deg'],-.075)
                cartesian_move(lambda:arm_q(left),send_left,left_mount,quat,point,1.5,'release outer panel')
                for _ in range(120 if both else 60):
                    tick('verify released door')

            if args.phase == 'close-probe':
                push_door_shut()
            picks = ('orange_key','blue_key') if both else {
                'pickup': ('blue_key',),'return-blue': ('blue_key',),'return-orange': ('orange_key',),
            }.get(args.phase,())
            for key_name in picks:
                right_side = both or key_name == 'orange_key'
                key_arm = robot if right_side else base.scene['left_robot']
                key_mount = mount if right_side else (0., .305, .75)
                arm_start = 7 if right_side else 0
                rotation_yaw = math.pi/2 if right_side else -math.pi/2
                pick_orientation = Rotation.from_matrix(grip_rotation(rotation_yaw))

                def send_key(q, phase):
                    action[arm_start:arm_start+6] = q
                    tick(phase)

                def move_key(point, seconds, phase, orientation=pick_orientation, orient=False):
                    cartesian_move(lambda: arm_q(key_arm), send_key, key_mount, orientation.as_quat(),
                                   point, seconds, phase, interpolate_orientation=orient)

                pose = np.array(trace[-1]['keys'][key_name]['pose_xyzw'])
                handle = np.array([0.,0.,-.044])
                fob = pose[:3]+Rotation.from_quat(pose[3:]).apply(handle)
                grasp = fob+[0.,0.,args.fob_offset]
                hover = grasp + [0., 0., .075]
                move_key(hover, 3., 'approach fob', orient=True)
                action[arm_start+6] = .55
                move_key(grasp, 2., 'descend to fob')
                action[arm_start+6] = 0.
                for _ in range(22):
                    tick('close on fob')
                move_key(hover, 2., 'lift fob')
                pose_held = np.array(trace[-1]['keys'][key_name]['pose_xyzw'])
                fob_lift = pose_held[:3]+Rotation.from_quat(pose_held[3:]).apply(handle)-fob
                fingers = slice(4,6) if right_side else slice(1,3)
                if fob_lift[2] < .05 or min(trace[-1]['keys'][key_name]['gripper_force_n'][fingers]) < .1:
                    raise RuntimeError(f'{key_name} did not lift with the gripper')
                if return_phase:
                    if trace[-1]['hinge_deg'] > -53:
                        raise RuntimeError('Door opening is insufficient for return path')
                    # Preserve observed relative pose, allowing the real gripper
                    # to carry the key as a freely simulated contact object.
                    tcp, gripper_matrix = ik.tcp_pose(arm_q(key_arm), key_mount)
                    wrist = Rotation.from_matrix(gripper_matrix)
                    rel_key = wrist.inv() * Rotation.from_quat(pose_held[3:])
                    offset = wrist.inv().apply(pose_held[:3] - tcp)
                    hook_y = HOOK_Y[key_name]+cabinet_y
                    waypoints = np.array([
                        [args.cabinet_x+.004, hook_y, 1.014],
                        [args.cabinet_x+.112, hook_y, 1.014],
                        [args.cabinet_x+.112, hook_y, .988],
                    ])
                    for yaw in (90., -90.):
                        target_wrist = Rotation.from_euler('z', yaw, degrees=True) * rel_key.inv()
                        points = waypoints - target_wrist.apply(offset)
                        seed = arm_q(key_arm)
                        residuals = []
                        for point in points:
                            seed, error, rotation_error = ik.solve(
                                point, target_wrist.as_quat(), seed, key_mount, iters=120)
                            residuals.append(error < .001 and rotation_error < math.radians(.3))
                        if all(residuals):
                            break
                    else:
                        raise RuntimeError('Neither vertical key orientation reaches the hook')
                    if not right_side:
                        move_key([.18, .16, .848], 2.5, 'round free edge')
                        move_key([.18, -.05, .848], 2.5, 'cross in front of door')
                    lane = [.42, cabinet_y-.14 if both else -.18 if right_side else -.055, .95]
                    move_key(lane, 3.5, 'turn key in clear approach lane', target_wrist, orient=True)
                    for i, point in enumerate(points):
                        move_key(point, 3., ('face hook', 'thread ring', 'lower ring')[i], target_wrist)
                    action[arm_start+6] = 1.
                    for _ in range(30):
                        tick('release ring')
                    clearance = tcp_world(key_arm) + [0., 0., -.018]
                    move_key(clearance, 1., 'clear below fob', target_wrist)
                    move_key(clearance + [-.065, 0., 0.], 2., 'withdraw from ring', target_wrist)
                    for _ in range(60):
                        tick('verify hanging after release')
                    if both or args.close_door:
                        move_key([.29,-.20,.93], 2.5, 'back out of cabinet', target_wrist)
                    if args.close_door or (both and key_name == picks[-1]):
                        push_door_shut()
        return {
            'phase': args.phase, 'steps': len(trace), 'settled_keys': still,
            'final_hinge_deg': trace[-1]['hinge_deg'],
            'max_opening_deg': max(-row['hinge_deg'] for row in trace),
            'door_released': max(sum(row['contacts_n']['door']) for row in trace[-20:]) < .1,
            'door_closed': max(abs(row['hinge_deg']) for row in trace[-20:]) <= 3.,
            'cabinet_x_m': args.cabinet_x, 'key_contact_offset_m': args.key_offset,
            'cabinet_y_m': cabinet_y,
            'physics_hz':480 if args.fine_contact else 240,
            'hooks_support_key': settled_hangs(trace, args.cabinet_x,cabinet_y=cabinet_y),
            'source_assets': sorted(path.name for path in args.assets.glob('*.usda')),
        }
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('settle', 'door', 'close-probe', 'support-control', 'pickup',
                                           'return-blue', 'return-orange', 'return-both'),
                        required=True)
    parser.add_argument('--cabinet-x', type=float, choices=(.55, .60, .65), default=.55)
    parser.add_argument('--open-degrees', type=float, default=70.)
    parser.add_argument('--key-offset', type=float, default=.0008)
    parser.add_argument('--fine-contact',action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--close-door', action='store_true',
                        help='Probe physical panel closure after the orange-key return.')
    parser.add_argument('--fob-offset', type=float, default=.009,
                        help='TCP height above the measured key fob centre in metres.')
    args = parser.parse_args()
    if args.close_door and args.phase != 'return-orange':
        parser.error('--close-door requires --phase return-orange')
    if args.dream and args.phase != 'return-both':
        parser.error('--dream requires --phase return-both')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        report = run(args)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    except Exception:
        (args.output/'failure.json').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
