"""Raise a friction-held blind carriage and tilt its linked slats by contact."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_blind_trial import HEIGHT, INSTRUCTION, ORIGIN, STEP_LIMIT, TILT, audit


MOUNTS = {'left':(0.,.305,.75),'right':(0.,-.305,.75)}
LIFT_TAB = np.array([-.131,.169,.064])
TILT_TAB = np.array([-.110,-.078,0.])


def run(args):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from PIL import Image
    import torch
    from yam_arena import ik, inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    output = args.output
    asset = args.asset or output/'blind.usda'
    if args.asset is None:
        from run_blind_linkage_probe import author

        author(asset,True,True,ORIGIN)
    bodies = {'rail':'/Carriage','tilt':'/Slat1'}

    def room(*_):
        return [scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.43,.44,.42)),
                scene._box('window_surround',(.032,.59,.70),(.535,.058,.76),(.78,.78,.74)),
                scene._box('glass_panel',(.021,.342,.350),(.504,.035,.825),(.45,.67,.74)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1425.))]

    install(asset,output,ORIGIN,(0.,0.,0.,1.),bodies,room_factory=room,robot_contact_offset=.0005)
    width,height = (320,176) if args.dream else (640,360)
    env = inspect_embodiment.YamArenaEmbodiment(headless=True,cam_width=width,cam_height=height,
        control_hz=20,object_jitter=0.,task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='raise-and-tilt-window-blind',instruction=INSTRUCTION)
        env.reset(specification,seed=0)
        env._task_step = lambda:SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        eye = [-.22,.035,1.12] if args.setup_only else [.02,-.58,1.15]
        camera.set_world_poses_from_view(torch.tensor([eye],device=base.device),
                                         torch.tensor([[.37,.035,.82]],device=base.device))
        base.sim.render()
        camera.update(.05,force_recompute=True)
        if args.setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        observer = MechanismObserver(base.sim.stage)
        robots = {side:base.scene[side+'_robot'] for side in MOUNTS}
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        down = Rotation.from_quat(DOWN_GRIP)
        wrists = {side:Rotation.from_euler('y',-12.,degrees=True)*down for side in MOUNTS}

        with (output/'trace.jsonl').open('w') as log, record_trial(output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('The blind trial exceeded its control limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                states = observer.read()['bodies']
                contacts = {}
                for name in bodies:
                    values = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    contacts[name] = np.linalg.norm(values.reshape(-1,6,3).sum(0),axis=1).tolist()
                poses = {key:states[path]['pose_xyzw'] for key,path in bodies.items()}
                angles = [float(Rotation.from_quat(states[f'/Slat{i}']['pose_xyzw'][3:]).as_euler('xyz',degrees=True)[1])
                          for i in range(3)]
                row = {'step':len(trace),'phase':phase,'lift_m':poses['rail'][2]-ORIGIN[2],
                       'slat_deg':angles,'pose_xyzw':poses,'rail_contact_n':contacts['rail'],
                       'slat_contact_n':contacts['tilt'],'action':action.tolist(),
                       'tcp_xyz_m':{side:tcp_world(robot).tolist() for side,robot in robots.items()}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                frame(observation.images['top_cam'])
                if len(trace)%25==0:
                    Image.fromarray(observation.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,round(row['lift_m'],4),np.round(angles,1),flush=True)
                return observation

            if args.dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,tick,audit,trace,action,STEP_LIMIT,[asset])

            def hold(count,phase):
                for _ in range(count):
                    tick(phase)

            def move(side,point,duration,phase,orient=False):
                offset = 0 if side=='left' else 7

                def send(q,label):
                    action[offset:offset+6] = q
                    tick(label)

                cartesian_move(lambda:arm_q(robots[side]),send,MOUNTS[side],wrists[side].as_quat(),point,duration,
                               phase,interpolate_orientation=orient)

            def world_point(name,local):
                pose = np.asarray(trace[-1]['pose_xyzw'][name])
                return pose[:3]+Rotation.from_quat(pose[3:]).apply(local)

            def align(side,point,phase):
                offset = 0 if side=='left' else 7
                start = np.copy(arm_q(robots[side]))
                target,pe,re = ik.solve(point,wrists[side].as_quat(),start,MOUNTS[side],iters=280)
                if pe>.001 or re>np.deg2rad(.3):
                    raise RuntimeError(f'{phase} overhead alignment failed: {pe:.4f} m, {re:.4f} rad')
                for alpha in np.linspace(0.,1.,55)[1:]:
                    action[offset:offset+6] = start+ik.smoothstep(alpha)*(target-start)
                    tick(phase)

            hold(35,'settle the open blind')
            center = world_point('rail',LIFT_TAB)+[0.,0.,.008]
            action[6] = .72
            align('left',[.185,.205,.866],'reach the open space in front of the blind')
            move('left',[.185,center[1],.840],2.35,'cross in front of the frame')
            move('left',[.206,center[1],center[2]+.055],1.55,'descend ahead of the slats')
            move('left',center+[0.,0.,.055],1.4,'approach the blue lower rail handle')
            move('left',center,1.55,'lower over the blue handle')
            action[6] = 0.
            hold(21,'grasp the blue handle')
            if min(trace[-1]['rail_contact_n'][1:3])<=.05:
                raise RuntimeError('The blue handle did not make bilateral contact')
            tcp = tcp_world(robots['left'])
            move('left',tcp+[0.,0.,HEIGHT],2.15,'raise the blind rail')
            hold(13,'check the stripe height')
            target = np.asarray(tcp)+[0.,0.,HEIGHT]
            for _ in range(12):
                error = HEIGHT-trace[-1]['lift_m']
                if abs(error)<=.003:
                    break
                target[2] += np.clip(.70*error,-.005,.005)
                move('left',target,.32,'align the lower rail to the stripe')
                hold(5,'check the lower rail height')
            action[6] = .78
            hold(16,'release the blue handle')
            move('left',tcp_world(robots['left'])+[-.016,0.,.048],1.4,'withdraw the left gripper')
            initial = np.copy(arm_q(robots['left']))
            for alpha in np.linspace(0.,1.,43)[1:]:
                action[:6] = initial+ik.smoothstep(alpha)*(np.asarray(READY)-initial)
                tick('park the left arm')

            direction = wrists['right'].apply([0.,0.,1.])
            action[13] = 0.
            pad = world_point('tilt',TILT_TAB)
            align('right',[.185,-.205,.905],'reach the open space beside the tilt lever')
            move('right',[.185,pad[1],pad[2]+.065],2.3,'cross in front of the frame to the lever')
            move('right',pad+[0.,0.,.065],1.45,'approach the orange tilt lever')
            move('right',pad-.012*direction,1.50,'contact the orange tilt lever')
            advance = 0.
            for _ in range(30):
                current = np.mean(trace[-1]['slat_deg'])
                if current<=TILT+1.05:
                    break
                pad = world_point('tilt',TILT_TAB)
                load = sum(trace[-1]['slat_contact_n'][3:])
                advance = float(np.clip(advance+(.0011 if load<2. else -.0005),.0008,.031))
                move('right',pad-.012*direction+[0.,0.,-advance],.34,'tilt all three slats down')
                hold(4,'check the three slat angles')
            move('right',tcp_world(robots['right'])+[0.,-.008,.055],1.55,'release the tilt lever')
            action[13] = 1.
            hold(42,'observe the released blind')
        return audit(trace)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--asset',type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--setup-only',action='store_true')
    modes.add_argument('--dream',action='store_true')
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
