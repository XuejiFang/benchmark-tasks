"""Use physical button strokes to program and arm a simulated bedside clock."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np

from audit_alarm_clock_trial import ClockInput, INSTRUCTION, audit


KEYS = {'Hour': (.046, .052), 'Minute': (.046, 0.), 'Alarm': (.046, -.052)}
SEGMENTS = ('abcdef', 'bc', 'abdeg', 'abcdg', 'bcfg', 'acdfg', 'acdefg', 'abc', 'abcdefg', 'abcdfg')
ORIGIN = (.302, .025, .75)


def author(destination):
    from pxr import Gf, UsdPhysics
    from build_key_return_scene import box, rigid, stage_at
    from build_tea_probe import contact

    stage = stage_at(destination, 'Clock')
    root = '/Clock'
    housing = root+'/Housing'
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    rigid(stage, housing, .27)
    box(stage, housing+'/Shell', (0., 0., .015), (.150, .158, .030), (.15, .17, .20))
    box(stage, housing+'/DisplayGlass', (-.029, 0., .031), (.055, .137, .003), (.026, .040, .036))
    mount = UsdPhysics.FixedJoint.Define(stage, root+'/Mount')
    mount.CreateBody1Rel().SetTargets([housing])

    def visual(path, xyz, size, rgb):
        box(stage, path, xyz, size, rgb)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Set(False)

    # The display reads left to right in decreasing Y when viewed from the front.
    shapes = {'a': (.010, 0., .002, .012), 'b': (.005, -.006, .009, .002),
              'c': (-.005, -.006, .009, .002), 'd': (-.010, 0., .002, .012),
              'e': (-.005, .006, .009, .002), 'f': (.005, .006, .009, .002),
              'g': (0., 0., .002, .012)}
    for digit, y in enumerate((.046, .023, -.023, -.046)):
        for segment, (dx, dy, sx, sy) in shapes.items():
            visual(f'{housing}/Display/Digit{digit}/{segment}',
                   (-.029+dx, y+dy, .0328), (sx, sy, .001), (.45, .90, .56))
    for index, x in enumerate((-.034, -.024)):
        visual(f'{housing}/Display/Colon{index}', (x, 0., .0328), (.002, .002, .001), (.45, .90, .56))
    visual(housing+'/Display/Armed', (-.016, -.062, .0328), (.006, .005, .001), (1., .54, .19))
    for name, (x, y) in KEYS.items():
        key = root+'/'+name
        rigid(stage, key, .008, (x, y, .035))
        color = {'Hour': (.39, .49, .58), 'Minute': (.52, .57, .62), 'Alarm': (.69, .37, .15)}[name]
        box(stage, key+'/ButtonCap', (0., 0., 0.), (.026, .026, .008), color)
        contact(stage.GetPrimAtPath(key+'/ButtonCap'), .00025)
        visual(key+'/Mark', (0., 0., .0041), (.002, .013, .0003), (.84, .88, .90))
        if name=='Hour':
            visual(key+'/Plus', (0., 0., .0041), (.012, .002, .0003), (.84, .88, .90))
        if name=='Alarm':
            visual(key+'/ArmMark', (0., 0., .0041), (.010, .006, .0003), (.84, .88, .90))
        joint = UsdPhysics.PrismaticJoint.Define(stage, root+'/Press'+name)
        joint.CreateAxisAttr('Z')
        joint.CreateBody0Rel().SetTargets([housing])
        joint.CreateBody1Rel().SetTargets([key])
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, .035))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        joint.CreateLowerLimitAttr(-.0032)
        joint.CreateUpperLimitAttr(0.)
        joint.CreateCollisionEnabledAttr(False)
        spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        spring.CreateTargetPositionAttr(0.)
        spring.CreateStiffnessAttr(255.)
        spring.CreateDampingAttr(2.8)
    stage.GetRootLayer().Save()


def run(output, *, dream=False, setup_only=False, fixture=None):
    from inspect_robots import Scene
    from isaaclab import sim as sim_utils
    from pxr import UsdGeom
    import torch
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move, feedback_press
    from record_trial import record_trial
    from yam_arena import inspect_embodiment, scene
    from yam_arena.constants import READY, PAD_TIP_PAST_TCP
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q
    from PIL import Image

    if fixture is None:
        asset = output/'clock.usda'
        author(asset)
    else:
        asset = fixture.resolve()
    paths = {name: '/'+name for name in KEYS}

    def room(*_):
        return [
            scene._box('nightstand', (.48, .50, .04), (.36, .05, .73), (.37, .29, .23)),
            scene._box('floor', (3., 3., .04), (.45, 0., -.02), (.44, .41, .42)),
            scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1450.)),
        ]

    install(asset, output, ORIGIN, (0., 0., 0., 1.), paths, room_factory=room, robot_contact_offset=.00045)
    height, width = (176, 320) if dream else (360, 640)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True, cam_height=height, cam_width=width, control_hz=20, object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='bedside-clock', instruction=INSTRUCTION)
        env.reset(specification, seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.06, -.02, 1.11]], device=base.device),
                                         torch.tensor([[.31, .02, .777]], device=base.device))
        observer = MechanismObserver(base.sim.stage)
        display = observer.root+'/Housing/Display'
        action = np.array([*READY, 0., *READY, 0.])
        rows = []
        clock = ClockInput()

        def refresh_display():
            hh, mm = divmod(clock.minute, 60)
            for index, number in enumerate(f'{hh:02d}{mm:02d}'):
                for segment in 'abcdefg':
                    image = UsdGeom.Imageable.Get(base.sim.stage, f'{display}/Digit{index}/{segment}')
                    attr = image.GetVisibilityAttr()
                    attr.Set('inherited' if segment in SEGMENTS[int(number)] else 'invisible')
            UsdGeom.Imageable.Get(base.sim.stage, display+'/Armed').GetVisibilityAttr().Set(
                'inherited' if clock.armed else 'invisible')

        refresh_display()
        base.sim.render()
        camera.update(.05,force_recompute=True)
        if setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope': 'fixed reset screenshot', 'seed': 0}
        with (output/'trace.jsonl').open('w') as raw, record_trial(output, True, width=width, height=height) as record:
            def step(phase, inference_latency=None):
                if len(rows)>=2400:
                    raise RuntimeError('Clock trial reached its control limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()
                travel, contact = {}, {}
                for key, path in paths.items():
                    hardware = base.scene.sensors['mechanism_contact_'+key]
                    force = array(hardware.data.normal_force_matrix_w).reshape(-1, 6, 3).sum(0)
                    contact[key] = float(np.linalg.norm(force, axis=-1).sum())
                    travel[key] = state['bodies']['/Housing']['pose_xyzw'][2]+.035-state['bodies'][path]['pose_xyzw'][2]
                if clock.observe(travel):
                    refresh_display()
                row = {'step': len(rows), 'phase': phase, 'action': action.tolist(), 'travel_m': travel,
                       'contact_n': contact, 'alarm_minute': clock.minute, 'armed': clock.armed}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                rows.append(row)
                raw.write(json.dumps(row, allow_nan=False)+'\n')
                raw.flush()
                record(observation.images['top_cam'])
                if len(rows)%25==0:
                    print('STEP', len(rows), phase, clock.minute, clock.armed, flush=True)
                if not dream and max(contact.values())>15.:
                    raise RuntimeError('Clock button force guard exceeded')
                return observation

            if dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,step,lambda _: audit(rows),
                                 rows,action,2400,[asset])

            def hold(n, phase):
                for _ in range(n):
                    step(phase)

            def press(key):
                robot = base.scene['left_robot']
                mount = (0., .305, .75)
                x, y = KEYS[key]
                point = np.array(ORIGIN)+[x, y, .039+PAD_TIP_PAST_TCP]

                def send(q, phase):
                    action[:6] = q
                    step(key+' '+phase)

                def move(target, seconds, phase, orient=False):
                    cartesian_move(lambda: arm_q(robot), send, mount, DOWN_GRIP, target, seconds, phase,
                                   interpolate_orientation=orient)

                def measure():
                    row = rows[-1]
                    return (row['travel_m'][key], row['contact_n'][key],
                            max(row['contact_n'][other] for other in KEYS if other!=key))

                hover = point+[0., 0., .040]
                move(hover, 2.3, 'approach', True)
                feedback_press(move, measure, point, [0., 0., -1.], .0022,
                               max_corrections=32, force_limit=12., neighbor_limit=.1)
                move(hover, 1.25, 'release')
                hold(12, key+' settle')

            hold(30, 'reset settle')
            press('Hour')
            press('Minute')
            press('Minute')
            press('Alarm')
            hold(28, 'check released clock')
        return {'scope': 'alarm state changes through physical clock button press and release'} | audit(rows)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dream', action='store_true')
    modes.add_argument('--setup-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        result = run(args.output,dream=args.dream,setup_only=args.setup_only,fixture=args.fixture)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
