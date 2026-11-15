"""Physically separate one nested bowl and place it on its breakfast mat."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import traceback
import time
from types import SimpleNamespace

import numpy as np


DREAM_INSTRUCTION = (
    'Separate the two stacked bowls. Put the teal bowl upright in the center of the teal mat '
    'and the cream bowl upright in the center of the tan mat. Place the dark-handled spoon '
    'on the bare tabletop beside the teal mat and the light-handled spoon beside the tan mat, '
    'each between the two mats with its head pointing toward the robot bases. '
    'Release all four objects and leave them resting on their places.'
)


def run(args):
    import isaaclab.sim as sim_utils
    import torch
    from PIL import Image
    from inspect_robots import Scene
    from scipy.spatial.transform import Rotation
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    from isaaclab.sensors import ContactSensorCfg
    from yam_arena import embodiment, ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP, grip_rotation
    from yam_arena.runtime import arm_q, tcp_world
    from yam_arena.robot import asset_link_paths
    from articulated_fixture import array
    from expert_motion import cartesian_move
    from record_trial import record_trial

    # Preserve the pose where the left arm verified upper-bowl placement and
    # bring the second pedestal closer instead of moving the nested stack.
    stack = (.41, .12)
    mat = np.array([.46, .37])
    other_mat = np.array([.46, -.37])
    spoon_starts = {'dark_spoon': (.26, .23), 'light_spoon': (.28, -.24)}
    spoon_start_yaw = {'dark_spoon': 75., 'light_spoon': 105.}
    spoon_destinations = {'dark_spoon': (.46, .235), 'light_spoon': (.46, -.235)}
    bowl_names = ('cream_bowl', 'teal_bowl')
    surface = .75
    mat_top = .754
    base_z = .83
    right_y = -.21 if args.full else -.305
    station.BASES = {'left': (0., .305, base_z), 'right': (0., right_y, base_z)}

    def room(*_):
        return [
            scene._box('floor', (3., 3., .04), (.45, 0., -.02), (.38, .41, .43)),
            scene._box('dining_table', (.68, 1., .045), (.46, 0., surface-.0225), (.31,.19,.12)),
            scene._box('teal_mat', (.23, .23, .004), (*mat, surface+.002), (.12, .27, .29)),
            scene._box('cream_mat', (.23, .23, .004), (*other_mat, surface+.002), (.54, .42, .28)),
            *[scene._box(side+'_pedestal', (.15,.15,base_z), (0.,y,base_z/2), (.30,.32,.33))
              for side,y in [('left',.305),('right',right_y)]],
            scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1100)),
        ]

    def bowl(spec):
        if spec.name in spoon_starts:
            position = (*spoon_starts[spec.name], surface+.002)
            rotation = tuple(map(float, Rotation.from_euler('z', spoon_start_yaw[spec.name], degrees=True).as_quat()))
        else:
            position = (*stack, surface + (.002 if spec.name == 'cream_bowl' else .018))
            rotation = (0., 0., 0., 1.)
        return Object(
            name=spec.name, object_type=ObjectType.RIGID,
            usd_path=str((args.assets/f'{spec.name}.usda').resolve()),
            initial_pose=Pose(position_xyz=position, rotation_xyzw=rotation),
            spawn_cfg_addon={
                'physics_material': scene._grippy(),
                'collision_props': sim_utils.CollisionPropertiesCfg(contact_offset=.0008, rest_offset=0.),
                'rigid_props': PhysxRigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                    max_depenetration_velocity=.25,
                ),
                'activate_contact_sensors': True,
            },
        )

    scene.task_room, scene.task_object = room, bowl
    original_build = scene.build_task_environment
    def build(*a, **kw):
        arena = original_build(*a, **kw)
        configure = arena.env_cfg_callback
        links = asset_link_paths()
        def enable_contact(cfg):
            cfg = configure(cfg)
            for name in (*bowl_names, *(spoon_starts if args.full else ())):
                setattr(cfg.scene, 'rim_contact_' + name, ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/' + name,
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/' + side + links[link]
                                            for side in ('LeftArm', 'RightArm')
                                            for link in ('gripper', 'tip_left', 'tip_right')],
                    history_length=0,
                ))
            return cfg
        arena.env_cfg_callback = enable_contact
        return arena
    scene.build_task_environment = build
    original_embodiment = embodiment.BimanualYamEmbodiment
    def make_embodiment(*a, **kw):
        return original_embodiment(*a, **(kw | {'robot_contact_offset': .001}))
    embodiment.BimanualYamEmbodiment = make_embodiment
    os.environ['ROBOCURVE_BREAKFAST_FULL'] = '1' if args.full else '0'
    width, height = (320, 176) if args.dream else (640, 360)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True, cam_height=height, cam_width=width, control_hz=20, object_jitter=0,
        task=str(Path(__file__).with_name('breakfast_fixture_task.py')),
    )
    try:
        scene_spec = Scene(id='breakfast-place-settings', instruction=DREAM_INSTRUCTION)
        env.reset(scene_spec, seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        camera = base.scene.sensors['top_cam']
        viewpoint = [[.45, -.03, 1.65]] if args.dream else [[.12, -.80, 1.44]]
        focus = [[.42, .01, .75]] if args.dream else [[.45, .23, .79]]
        camera.set_world_poses_from_view(torch.tensor(viewpoint, device=base.device),
                                         torch.tensor(focus, device=base.device))
        if args.dream:
            base.sim.render()
            camera.update(.05, force_recompute=True)
        robot = base.scene['left_robot']
        mount = (0., .305, base_z)
        arm_start = 0
        action = np.array([*READY, 1., *READY, 1.])
        trace = []
        with (args.output/'trace.jsonl').open('w') as out, record_trial(
                args.output, args.video, width=width, height=height) as write_frame:
            def tick(phase, inference_latency=None):
                obs = env.step(SimpleNamespace(data=action)).observation
                if write_frame is not None:
                    write_frame(obs.images['top_cam'])
                states = {}
                def gripper_forces(name):
                    force = base.scene.sensors['rim_contact_'+name].data.normal_force_matrix_w
                    return np.linalg.norm(array(force).reshape(-1, 6, 3).sum(0),axis=1).tolist()

                for name in bowl_names:
                    data = base.scene.rigid_objects[name].data
                    force = gripper_forces(name)
                    states[name] = {'pose_xyzw': array(data.root_pose_w)[0].tolist(),
                                    'velocity': array(data.root_vel_w)[0].tolist(),
                                    'finger_force_n': force[1:3], 'gripper_force_n': force}
                spoons = {}
                for name in spoon_starts if args.full else ():
                    data = base.scene.rigid_objects[name].data
                    spoons[name] = {'pose_xyzw': array(data.root_pose_w)[0].tolist(),
                                    'gripper_force_n': gripper_forces(name)}
                row = {'step': len(trace), 'phase': phase, 'action': action.tolist(),
                       'bowls': states, 'spoons': spoons}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                out.write(json.dumps(row, allow_nan=False)+'\n')
                out.flush()
                trace.append(row)
                if len(trace) in (1, 70) or len(trace)%25 == 0:
                    Image.fromarray(np.asarray(obs.images['top_cam'])).save(args.output/'latest.png')
                    print('STEP', len(trace), phase, flush=True)
                return obs

            if args.dream:
                from inspect_robots_dream.policy import DreamPolicy
                from audit_breakfast_fixture import score_two_settings

                policy = DreamPolicy(timeout_s=120, record=False)
                policy.bind(env.info)
                policy.reset(scene_spec)
                metadata = {
                    'policy': 'inspect_robots_dream.policy.DreamPolicy',
                    'model': policy.info.checkpoint, 'seed': 0, 'control_hz': 20,
                    'instruction': DREAM_INSTRUCTION, 'max_steps': args.max_steps,
                    'observation_size': [width, height], 'privileged_state_to_policy': False,
                    'versions': {name: importlib.metadata.version(name) for name in
                                 ('inspect-robots', 'inspect-robots-dream', 'dreamlabs-client')},
                    'asset_sha256': {name: hashlib.sha256((args.assets/name).read_bytes()).hexdigest()
                                     for name in sorted(f'{obj}.usda' for obj in (*bowl_names, *spoon_starts))},
                }
                (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
                observation = env._observation()
                for name, frame in observation.images.items():
                    Image.fromarray(frame).save(args.output/f'initial_{name}.png')
                started = time.monotonic()
                try:
                    for _ in range(args.max_steps):
                        chunk = policy.act(observation)
                        if len(chunk.actions) != 1:
                            raise RuntimeError('Expected one Dream action per control step')
                        values = np.asarray(chunk.actions[0].data, dtype=float)
                        if values.shape != (14,) or not np.isfinite(values).all():
                            raise RuntimeError('Nonfinite or invalid Dream action')
                        action[:] = values
                        observation = tick('dream control', chunk.inference_latency_s)
                        report = score_two_settings(trace[-20:])
                        if report['score'] == 1.:
                            break
                finally:
                    (args.output/'dream_transcript.json').write_text(json.dumps(policy.transcript(), indent=2)+'\n')
                    policy.close()
                return report | {'status': 'completed', 'steps': len(trace), 'kind': 'dream',
                                 'duration_s': len(trace)/20, 'wall_time_s': time.monotonic()-started}

            def send(q, phase):
                action[arm_start:arm_start+6] = q
                tick(phase)

            def move(point, seconds, phase, orient=False, quat=DOWN_GRIP):
                cartesian_move(lambda: arm_q(robot), send, mount, quat,
                               point, seconds, phase, interpolate_orientation=orient)

            for _ in range(70):
                tick('nested settle')
            initial = trace[-1]['bowls']
            center = np.array(initial['teal_bowl']['pose_xyzw'][:3])
            grasp = center + [0., .077, .046 + args.pad_clearance]
            raised = grasp + [0., 0., .115]
            move(grasp + [0., 0., .066], 3., 'approach upper rim', True)
            action[6] = .30
            move(grasp, 1.8, 'descend inside and outside upper rim')
            action[6] = 0.
            for _ in range(25):
                tick('pinch rim')
            move(raised, 2., 'lift upper bowl')
            picked = trace[-1]['bowls']
            rise = picked['teal_bowl']['pose_xyzw'][2] - initial['teal_bowl']['pose_xyzw'][2]
            if rise < .040:
                raise RuntimeError(f'Upper bowl did not separate: lift observed {rise:.3f} m')
            if abs(picked['cream_bowl']['pose_xyzw'][2]-initial['cream_bowl']['pose_xyzw'][2]) > .008:
                raise RuntimeError('Lower bowl lifted along with upper bowl')
            target = tcp_world(robot).copy()
            target[:2] += mat-np.asarray(picked['teal_bowl']['pose_xyzw'][:2])
            # The bowl rests on its opposite edge before it rotates flat.
            # Lead the nominal target by the observed centre displacement.
            target[1] -= .012
            move(target, 2., 'carry to mat')
            target[2] = mat_top + .046 + args.pad_clearance + .003
            move(target, 1.8, 'lower onto mat')
            action[6] = 1.
            for _ in range(20):
                tick('release onto mat')
            move(target + [0., 0., .05], 1.6, 'clear rim')
            for _ in range(30):
                tick('placed settle')
            if args.full:
                # Mirror the open-space rim contact for the lower bowl after
                # the first bowl has been removed.
                robot, mount, arm_start = base.scene['right_robot'], (0., right_y, base_z), 7
                cream = np.array(trace[-1]['bowls']['cream_bowl']['pose_xyzw'][:3])
                pinch = cream + [0., -.077, .046+args.pad_clearance]
                move(pinch + [0., 0., .065], 3.5, 'approach lower bowl', True)
                action[13] = .30
                move(pinch, 2., 'descend lower bowl')
                action[13] = 0.
                for _ in range(25):
                    tick('pinch lower bowl')
                move(pinch + [0., 0., .115], 2.5, 'lift lower bowl')
                carried = np.array(trace[-1]['bowls']['cream_bowl']['pose_xyzw'][:3])
                if carried[2]-cream[2] < .04:
                    raise RuntimeError('Lower bowl did not lift after the stack was separated')
                target = tcp_world(robot).copy()
                target[:2] += other_mat-carried[:2]
                target[1] += .005
                target[2] = min(target[2], .879)
                move(target, 2.5, 'carry lower bowl')
                target[2] = mat_top + .046 + args.pad_clearance + .003
                move(target, 2., 'lower lower bowl onto mat')
                action[13] = 1.
                for _ in range(30):
                    tick('release lower bowl')
                move(target+[0., 0., .05], 2., 'clear lower bowl')

                for name, side in [('dark_spoon','left'), ('light_spoon','right')]:
                    arm_start = 0 if side == 'left' else 7
                    grip_idx = arm_start+6
                    mount = (0., .305 if side == 'left' else right_y, base_z)
                    robot = base.scene[side+'_robot']
                    start_pose = np.array(trace[-1]['spoons'][name]['pose_xyzw'])
                    start_rotation = Rotation.from_quat(start_pose[3:])
                    grasp = start_pose[:3] + start_rotation.apply([0., -.015, 0.])
                    grasp[2] = surface + .0135
                    head = start_rotation.apply([0., 1., 0.])
                    pick_yaw = math.atan2(head[1], head[0]) - math.pi
                    pick_quat = ik.matrix_to_quat_xyzw(grip_rotation(pick_yaw))
                    move(grasp+[0., 0., .059], 3., 'approach '+name, True, pick_quat)
                    action[grip_idx] = .40
                    move(grasp, 2., 'descend '+name, quat=pick_quat)
                    action[grip_idx] = 0.
                    for _ in range(20):
                        tick('pinch '+name)
                    move(grasp+[0., 0., .065], 2., 'lift '+name, quat=pick_quat)
                    held = np.array(trace[-1]['spoons'][name]['pose_xyzw'])
                    if held[2]-start_pose[2] < .035:
                        raise RuntimeError(name+' did not lift')
                    tcp, matrix = ik.tcp_pose(arm_q(robot), mount)
                    wrist = Rotation.from_matrix(matrix)
                    inverse = wrist.inv()
                    spoon_orientation = Rotation.from_euler('z', 90., degrees=True)
                    place = spoon_orientation * (inverse * Rotation.from_quat(held[3:])).inv()
                    offset = inverse.apply(held[:3] - tcp)
                    xyz = np.array([*spoon_destinations[name], surface+.002]) - place.apply(offset)
                    move(xyz+[0., 0., .045], 3., 'rotate and carry '+name, True, place.as_quat())
                    move(xyz, 2., 'lower '+name, quat=place.as_quat())
                    action[grip_idx] = 1.
                    for _ in range(23):
                        tick('release '+name)
                    move(xyz+[0., 0., .054], 2., 'clear '+name, quat=place.as_quat())
                for _ in range(30):
                    tick('both settings settle')
        final = np.array(trace[-1]['bowls']['teal_bowl']['pose_xyzw'][:3])
        return {
            'steps': len(trace), 'bowl_lift_m': float(rise),
            'final_teal_position_m': final.tolist(), 'mat_xy_m': mat.tolist(),
            'placed_on_mat': bool(np.linalg.norm(final[:2] - mat) < .026 and
                                  abs(final[2] - mat_top) < .008),
            'full_trial': args.full, 'final_objects': trace[-1],
        }
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pad-clearance', type=float, default=.004)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--dream', action='store_true')
    parser.add_argument('--max-steps', type=int, default=2400)
    args = parser.parse_args()
    if args.dream and (not args.full or not 33 <= args.max_steps <= 2400):
        parser.error('Dream requires --full and 33..2400 control steps')
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
