"""Open an anchored reading book and turn its physical marked leaf using YAM."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_reading_trial import INSTRUCTION, ORIGIN, STEP_LIMIT, audit


TABS = {'Cover': np.array([.151, -.072, .006]), 'Page': np.array([.143, .056, 0.])}


def run(output,movable_book=False,fixture=None,dream=False,setup_only=False):
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

    asset = fixture.resolve() if fixture is not None else output/'book_contact.usda'
    if fixture is None:
        from run_book_hinge_probe import author

        author(asset,.00018,True,True,anchor_in_stand=not movable_book,
               book_offset=(.185,0.,-.028) if movable_book else None)
    panels = {name: '/Book/'+name for name in TABS}
    bodies = panels | ({'Spine': '/Book/Spine'} if movable_book else {})

    def room(*_):
        return [
            scene._box('reading_table', (.58, .87 if movable_book else .62, .040),
                       (.38, .080 if movable_book else .045, .662), (.34, .24, .17)),
            scene._box('floor', (3., 3., .040), (.45, 0., -.020), (.46, .43, .39)),
            scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1400.)),
        ]

    install(asset, output, ORIGIN, (0., 0., 0., 1.), bodies, room_factory=room, robot_contact_offset=.0005)
    height,width = (176,320) if dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True, cam_height=height,cam_width=width,control_hz=20,object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='marked-book-contact',instruction=INSTRUCTION if movable_book else
                              'Open the book and turn the blue marked leaf.')
        env.reset(specification,seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.57, -.37, 1.16]], device=base.device),
                                         torch.tensor([[.25, .055, .75]], device=base.device))
        observer = MechanismObserver(base.sim.stage)
        action = np.array([*READY, 1., *READY, 1.])
        trace = []
        down = Rotation.from_quat(DOWN_GRIP)

        if setup_only or dream:
            base.sim.render()
            camera.update(.05,force_recompute=True)
        if setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}

        with (output/'trace.jsonl').open('w') as raw, record_trial(output,True,width=width,height=height) as record:
            def step(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('Reading task exceeded the control limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()
                book = state['articulations']['/Book']
                contacts = {}
                for name in bodies:
                    sensor = base.scene.sensors['mechanism_contact_'+name]
                    force = array(sensor.data.normal_force_matrix_w).reshape(-1, 6, 3).sum(0)
                    contacts[name] = np.linalg.norm(force, axis=-1).tolist()
                row = {'step': len(trace), 'phase': phase, 'action': action.tolist(),
                       'angles_deg': dict(zip(book['names'], np.rad2deg(book['positions']).tolist())),
                       'contacts_n': contacts,
                       'poses': {name: state['bodies'][path]['pose_xyzw'] for name, path in bodies.items()}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                raw.write(json.dumps(row, allow_nan=False)+'\n')
                raw.flush()
                record(observation.images['top_cam'])
                if len(trace) % 25 == 0:
                    print('STEP', len(trace), phase, row['angles_deg'], flush=True)
                if not dream and any(sum(values)>(215. if name=='Spine' else 115.) for name,values in contacts.items()):
                    raise RuntimeError('Book panel contact exceeded the force guard')
                return observation

            if dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,step,audit,
                                 trace,action,STEP_LIMIT,[asset])

            def hold(controls, phase):
                for _ in range(controls):
                    step(phase)

            def park(*sides):
                from yam_arena.ik import smoothstep

                starts = {side:arm_q(base.scene[side+'_robot']).copy() for side in sides}
                for i in range(1,43):
                    fraction = smoothstep(i/42)
                    for side,begin in starts.items():
                        index = 0 if side=='left' else 7
                        action[index:index+6] = begin+fraction*(np.asarray(READY)-begin)
                    step('park '+(' and '.join(sides))+' arm')

            def carry_book():
                arms = [base.scene[side+'_robot'] for side in ('left','right')]
                mounts = [(0.,.305,.75),(0.,-.305,.75)]
                start = np.asarray(trace[-1]['poses']['Spine'])
                orientation = Rotation.from_quat(start[3:])
                wrists = [Rotation.from_euler('z',90.,degrees=True)*down for _ in arms]
                pickup = start[:3]+orientation.apply([[.021,.121,.033],[.021,-.121,.033]])

                def move_pair(points,seconds,phase,orient=False):
                    plans = []
                    for arm,mount,wrist,point in zip(arms,mounts,wrists,points):
                        plan = []
                        cartesian_move(lambda:arm_q(arm),lambda q,_:plan.append(q.copy()),mount,
                                       wrist.as_quat(),point,seconds,phase,interpolate_orientation=orient)
                        plans.append(plan)
                    for left,right in zip(*plans):
                        action[:6] = left
                        action[7:13] = right
                        step(phase)

                action[[6,13]] = .58
                move_pair(pickup+[0.,0.,.037],2.8,'approach both book tabs',True)
                hold(20,'let both arms reach the approach pose')
                move_pair(pickup,1.65,'enter opposing book grasps',True)
                hold(13,'settle open fingertips around both tabs')
                action[[6,13]] = 0.
                hold(22,'grasp the book on both sides')
                move_pair(pickup+[0.,0.,.055],2.1,'lift the book with both arms',True)
                carried = np.asarray(trace[-1]['poses']['Spine'])
                corners = np.array([[x,y,0.] for x in (-.020,.148) for y in (-.113,.133)])
                support_z = (Rotation.from_quat(carried[3:]).apply(corners)+carried[:3])[:,2].min()
                forces = trace[-1]['contacts_n']['Spine']
                if support_z<ORIGIN[2]+.004 or min(sum(forces[1:3]),sum(forces[4:]))<=.1:
                    raise RuntimeError('Both grippers did not lift the book clear of the stand')
                alignment = Rotation.from_quat(carried[3:]).inv()
                wrists = [alignment*wrist for wrist in wrists]
                offset = alignment.apply(np.asarray([tcp_world(arm) for arm in arms])-carried[:3])
                destination = ORIGIN+offset
                move_pair(destination+[0.,0.,.045],2.45,'carry the book over the reading stand',True)
                hold(16,'settle the book above the stand')
                for _ in range(3):
                    held = np.asarray(trace[-1]['poses']['Spine'][:3])
                    correction = ORIGIN[:2]-held[:2]
                    if np.linalg.norm(correction)<.002:
                        break
                    shift = np.r_[np.clip(correction,-.010,.010),0.]
                    move_pair(np.asarray([tcp_world(arm) for arm in arms])+shift,.55,'center the book above its guides')
                    hold(8,'settle centered book')
                held = np.asarray(trace[-1]['poses']['Spine'][:3])
                destination = np.asarray([tcp_world(arm) for arm in arms])+ORIGIN-held
                move_pair(destination,2.1,'lower the spine between the stand guides')
                hold(15, 'let the book rest on the stand')
                action[[6,13]] = .57
                hold(16, 'release book onto the stand')
                move_pair(destination+[0.,0.,.065],1.65,'withdraw both arms from the stand')
                action[[6,13]] = 1.
                hold(18, 'settle book in the stand')
                if np.linalg.norm(np.asarray(trace[-1]['poses']['Spine'][:3])-ORIGIN)>.012:
                    raise RuntimeError('The released book missed the stand')
                park('left','right')

            def turn(name, side, joint):
                robot = base.scene[side+'_robot']
                index = 7 if side=='right' else 0
                mount = (0., -.305 if side=='right' else .305, .75)
                pose = np.asarray(trace[-1]['poses'][name])
                grip = pose[:3]+Rotation.from_quat(pose[3:]).apply(TABS[name]+[0., 0., .008])

                def send(q, phase):
                    action[index:index+6] = q
                    step(phase)

                def move(point, wrist, seconds, phase, orient=False):
                    cartesian_move(lambda: arm_q(robot), send, mount, wrist.as_quat(), point, seconds,
                                   phase, interpolate_orientation=orient)

                action[index+6] = .65
                move(grip+[0., 0., .045], down, 2.5, 'approach '+name, True)
                move(grip, down, 1.5, 'lower over '+name)
                action[index+6] = 0.
                hold(18, 'grasp '+name)
                finger = 4 if side=='right' else 1
                if min(trace[-1]['contacts_n'][name][finger:finger+2]) <= .1:
                    raise RuntimeError(f'{name} did not reach both fingertips')
                pressed = np.asarray(trace[-1]['poses'][name])
                offset = tcp_world(robot)-pressed[:3]
                local_offset = Rotation.from_quat(pressed[3:]).inv().apply(offset)
                for angle in np.linspace(-9., -119., 14):
                    arc = Rotation.from_euler('y', angle, degrees=True)
                    if movable_book:
                        observed = trace[-1]['poses'][name]
                        spine = Rotation.from_quat(trace[-1]['poses']['Spine'][3:])
                        waypoint = observed[:3]+(spine*arc).apply(local_offset)
                    else:
                        waypoint = pressed[:3]+arc.apply(offset)
                    move(waypoint,
                         Rotation.from_euler('y', .65*angle, degrees=True)*down,
                         .31, 'turn '+name, True)
                if trace[-1]['angles_deg'][joint] > -99.:
                    raise RuntimeError(f'{name} did not pass vertical before release')
                action[index+6] = 1.
                hold(15, 'let go of '+name)
                move(tcp_world(robot)+[0., 0., .054], down, 1.6, 'withdraw from '+name, True)
                hold(30, 'settle '+name)

            hold(30, 'mechanism settle')
            if movable_book:
                carry_book()
            turn('Cover', 'right', 'CoverFold')
            cover = max(r['angles_deg']['CoverFold'] for r in trace[-20:])
            untouched = min(r['angles_deg']['LeafFold'] for r in trace[-20:])
            if cover > -150. or untouched < -12.:
                raise RuntimeError('Released cover and untouched leaf did not hold their positions')
            if movable_book:
                park('right')
            turn('Page', 'left', 'LeafFold')
        if movable_book:
            return {'scope':'free book transport and page turn'} | audit(trace)
        final = {name: max(r['angles_deg'][name] for r in trace[-20:]) for name in ('CoverFold', 'LeafFold')}
        contact = max(max(sum(force) for force in r['contacts_n'].values()) for r in trace[-20:])
        return {'scope':'anchored book contact test','controls':len(trace),
                'final_angles_deg':final,'max_final_contact_n':contact,
                'passed':max(final.values())<-150. and contact<=.1}
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--movable-book', action='store_true')
    parser.add_argument('--fixture',type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dream',action='store_true')
    modes.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    args.movable_book |= args.dream or args.setup_only or args.fixture is not None
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        result = run(args.output,args.movable_book,args.fixture,args.dream,args.setup_only)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
