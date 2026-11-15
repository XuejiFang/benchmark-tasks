"""Close a lid latch using ordinary YAM joint commands."""

import argparse
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def close_latch(side,robot,mount,observe,send,*,top_push=True):
    """Track the observed hinge and push its physical latch with one arm."""
    from yam_arena import ik
    from yam_arena.runtime import arm_q,tcp_world
    from expert_motion import cartesian_move

    axis = Rotation.from_euler('z',{'left':-90.,'right':90.}.get(side,0.),degrees=True)
    lateral = side in ('left','right')
    rear = side=='back'
    rear_yaw = 222.

    def wrist_at(angle):
        if rear:
            return (Rotation.from_euler('z',rear_yaw,degrees=True)
                    *Rotation.from_euler('y',min(176.,152.+.7*angle),degrees=True)
                    *Rotation.from_euler('z',-90.,degrees=True))
        if lateral:
            return axis*Rotation.from_euler('y',104.+angle,degrees=True)*Rotation.from_euler('z',-90.,degrees=True)
        if top_push:
            return Rotation.from_euler('y',104.+angle,degrees=True)
        return Rotation.from_euler('z',15.,degrees=True)*Rotation.from_euler('y',100.,degrees=True)

    def move(point,seconds,phase,orient=False):
        cartesian_move(lambda:arm_q(robot),send,mount,wrist.as_quat(),point,seconds,
                       side+' latch '+phase,interpolate_orientation=orient)

    def tab_point():
        _, pose = observe()
        face = -.0052 if top_push or rear or lateral else -.0038
        return pose[:3]+Rotation.from_quat(pose[3:]).apply([0.,face,-.013])

    initial_angle,_ = observe()
    wrist = wrist_at(initial_angle)
    forward = wrist.apply([0.,0.,1.])
    if rear and initial_angle<35.:
        # The loose lid can settle a few millimetres from the isolated fixture.
        # Validate the whole approach against the current lid and wrist before moving.
        start = arm_q(robot).copy()
        contact = tab_point()
        approach = None
        for rear_yaw in (222.,228.,234.,216.,240.):
            wrist = wrist_at(initial_angle)
            forward = wrist.apply([0.,0.,1.])
            for clearance in (.046,.039,.032):
                commands = []
                try:
                    cartesian_move(lambda:start,lambda q,_:commands.append(q.copy()),mount,
                                   wrist.as_quat(),contact-clearance*forward,2.8,'back latch approach',
                                   interpolate_orientation=True)
                    cartesian_move(lambda:commands[-1],lambda *_:None,mount,wrist.as_quat(),
                                   contact-.014*forward,1.5,'back latch touch')
                except RuntimeError:
                    continue
                approach = commands
                break
            if approach is not None:
                break
        if approach is None:
            raise RuntimeError('No reachable approach to the partially closed rear latch')
        for q in approach:
            send(q,'back latch approach')
    else:
        move(tab_point()-.055*forward,2.8,'approach',True)
    move(tab_point()-.014*forward,1.5,'touch')
    best,stalled,added_push = initial_angle,0,0.
    for _ in range(400 if rear else 320 if top_push or lateral else 145):
        measured,_ = observe()
        if measured<=(2.5 if rear else 2.):
            break
        wrist = wrist_at(measured)
        forward = wrist.apply([0.,0.,1.])
        near_closed = (rear and measured<12.) or (side=='right' and measured<25.)
        if measured<best-.8:
            best,stalled = measured,0
        else:
            stalled += 1
        if stalled==16:
            added_push = min(.006,added_push+.0015)
            stalled = 0
        stand_off = max(.0015,(.005 if near_closed else .008)-added_push)
        q,pe,re = ik.solve(tab_point()-stand_off*forward,wrist.as_quat(),
                          arm_q(robot),base_pos=mount,iters=100)
        if pe>.001 or re>math.radians(.3):
            raise RuntimeError(f'{side} latch push exceeds wrist limits')
        send(q,side+' latch push')
    if rear:
        wrist = wrist_at(16.)
        move(tcp_world(robot)+[-.006,.004,.022],1.7,'release',True)
    elif lateral:
        wrist = wrist_at(27.)
        move(tcp_world(robot)+axis.apply([-.006,0.,.025]),1.7,'release',True)
    elif top_push:
        clear = tcp_world(robot)+[-.007,0.,.002]
        move(clear,.8,'release')
        wrist = wrist_at(31.)
        move(clear+[-.020,0.,.055],1.8,'lift clear',True)
    else:
        move(tcp_world(robot)-.06*forward,1.7,'withdraw')


def run(args):
    from PIL import Image
    import torch
    from inspect_robots import Scene
    import isaaclab.sim as sim_utils
    from yam_arena import inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.runtime import tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from record_trial import record_trial

    assembly = '/Container/Geometry/Lid'
    side = args.side or ('back' if args.rear else 'front')
    use_right = side == 'right'
    joints = slice(7,13) if use_right else slice(0,6)
    tested = {'tab':assembly+'/Latch_'+side,'lid':assembly,'tub':'/Container/Geometry/Box'}

    def room(*_):
        return [scene._box('counter',(.64,.70,.04),(.42,.05,.73),(.38,.30,.24)),
                scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.40,.42,.43)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1400.))]

    install(args.usd,args.output,(0.,0.,0.),(0.,0.,0.,1.),tested,room_factory=room,robot_contact_offset=.0009)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True,cam_height=360,cam_width=640,control_hz=20,object_jitter=0,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        env.reset(Scene(id='lunchbox-latch-contact',instruction='Press the open latch until closed.'),seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        robot = base.scene['right_robot' if use_right else 'left_robot']
        mount = (0.,-.305 if use_right else .305,.75)
        camera = base.scene.sensors['top_cam']
        eye,look = ((.65,-.15,1.12),(.30,.08,.80)) if args.rear or args.side else ((.14,-.035,1.12),(.35,.245,.80))
        camera.set_world_poses_from_view(torch.tensor([eye],device=base.device),
                                         torch.tensor([look],device=base.device))
        mechanism = MechanismObserver(base.sim.stage)
        action = np.array([*READY,float(use_right),*READY,float(not use_right)])
        trace = []
        with (args.output/'trace.jsonl').open('w') as tracing, record_trial(args.output,True) as frame:
            def tick(phase):
                obs = env.step(SimpleNamespace(data=action)).observation
                state = mechanism.read()
                joint = state['articulations'][assembly]
                angle = float(np.rad2deg(joint['positions'][joint['names'].index('latch_'+side)]))
                contacts = {}
                for name in tested:
                    force = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    contacts[name] = np.linalg.norm(force.reshape(-1,6,3).sum(0),axis=1).tolist()
                row = {'phase':phase,'step':len(trace),'action':action.tolist(),'latch_deg':angle,
                       'contacts_n':contacts,'tcp_m':tcp_world(robot).tolist(),**state}
                tracing.write(json.dumps(row,allow_nan=False)+'\n')
                tracing.flush()
                trace.append(row)
                frame(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%20==0:
                    Image.fromarray(obs.images['top_cam']).save(args.output/'latest.png')
                    print('STEP',len(trace),phase,round(angle,2),flush=True)
                if sum(contacts['tub'])>15. or sum(contacts['tab'])>60. or sum(contacts['lid'])>30.:
                    raise RuntimeError('Unexpected latch fixture contact force')

            def send(q, phase):
                action[joints] = q
                tick(phase)

            def observe():
                row = trace[-1]
                return row['latch_deg'],np.asarray(row['bodies'][tested['tab']]['pose_xyzw'])

            for _ in range(45):
                tick('free hinge settle')
            initial_angle,_ = observe()
            if initial_angle < 10.:
                raise RuntimeError('Latch closed without contact during fixture settle')
            close_latch(side,robot,mount,observe,send,top_push=args.top_push)
            for _ in range(35):
                tick('released latch settle')
        return {'scope':'fixture held tub, latch moved by physical contact','side':side,'steps':len(trace),
                'after_free_settle_deg':initial_angle,'final_latch_deg':trace[-1]['latch_deg'],
                'peak_tab_contact_n':max(sum(r['contacts_n']['tab']) for r in trace),
                'max_final_tab_contact_n':max(sum(r['contacts_n']['tab']) for r in trace[-20:]),
                'max_final_angle_deg':max(r['latch_deg'] for r in trace[-20:])}
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--top-push',action='store_true')
    parser.add_argument('--rear',action='store_true')
    parser.add_argument('--side',choices=('left','right'))
    args = parser.parse_args()
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
