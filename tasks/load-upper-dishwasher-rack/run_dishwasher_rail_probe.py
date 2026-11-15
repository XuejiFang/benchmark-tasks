"""Pull and return the source dishwasher's upper rack using the YAM gripper."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from audit_dishwasher_trial import INSTRUCTION, RACK_SITES, RUN_LIMIT, WOOD_START, audit


RACK = '/Links/Upper_Rack_Assembly'
DOOR = '/Links/Door'
BASE = '/Links/Base'
PLACEMENT_X = .68
TAB_SOURCE_XYZ = (.265,.241,.614)
SOURCE_SHA256 = 'abe1873440af6f9b5f092bb8df7f81d6ae89115bd4744b54e1de531cecc90520'


def build_fixture(source,output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Expected inspected dishwasher source')
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/DishwasherFixture').GetPrim())
    root = '/DishwasherFixture/Machine'
    machine = UsdGeom.Xform.Define(stage,root)
    machine.GetPrim().GetReferences().AddReference(str(source.resolve()))
    machine.AddTranslateOp().Set(Gf.Vec3d(PLACEMENT_X,0.,0.))
    machine.AddRotateZOp().Set(180.)
    stage.GetPrimAtPath(root+'/PhysicsScene').SetActive(False)
    tub = UsdGeom.Xformable(stage.GetPrimAtPath(root+BASE))
    base_pose = tub.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    anchor = UsdPhysics.FixedJoint.Define(stage,'/DishwasherFixture/Mount')
    anchor.CreateBody1Rel().SetTargets([root+BASE])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(base_pose.ExtractTranslation()))
    anchor.CreateLocalRot0Attr(Gf.Quatf(base_pose.ExtractRotationQuat()))

    # Start the door at its horizontal stop. The upper rack still uses the
    # original freely moving rail and the original roller mimic joint.
    hinge = UsdPhysics.RevoluteJoint.Get(stage,root+'/Joints/Door_joint')
    local0 = Gf.Rotation(Gf.Quatd(hinge.GetLocalRot0Attr().Get()))
    local1 = Gf.Rotation(Gf.Quatd(hinge.GetLocalRot1Attr().Get()))
    world_rotation = base_pose.ExtractRotation()*local0*Gf.Rotation(Gf.Vec3d(1,0,0),90.)*local1.GetInverse()
    world_pivot = base_pose.Transform(Gf.Vec3d(hinge.GetLocalPos0Attr().Get()))
    world_center = world_pivot-world_rotation.TransformDir(Gf.Vec3d(hinge.GetLocalPos1Attr().Get()))
    world_door = Gf.Matrix4d(1.).SetRotate(world_rotation).SetTranslateOnly(world_center)
    door = UsdGeom.Xformable.Get(stage,root+DOOR)
    parent_inverse = door.ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    door.MakeMatrixXform().Set(world_door*parent_inverse)
    hinge.GetLowerLimitAttr().Set(89.95)
    hinge.GetUpperLimitAttr().Set(90.00002)
    hinge.GetPrim().CreateAttribute('state:angular:physics:position',Sdf.ValueTypeNames.Float).Set(90.)

    rack = UsdGeom.Xformable.Get(stage,root+RACK)
    rack_inverse = rack.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    tab_x,tab_y,tab_z = TAB_SOURCE_XYZ
    center = rack_inverse.Transform(Gf.Vec3d(PLACEMENT_X-tab_x,-tab_y,tab_z))
    handle = UsdGeom.Cube.Define(stage,root+RACK+'/ClipHandle')
    handle.CreateSizeAttr(1.)
    handle.AddTranslateOp().Set(center)
    handle.AddScaleOp().Set(Gf.Vec3f(.032,.017,.021))
    handle.CreateDisplayColorAttr([Gf.Vec3f(.72,.34,.09)])
    UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
    stage.GetRootLayer().Save()
    return np.array(center)


def run(args):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import torch
    from yam_arena import inspect_embodiment,scene,station
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q,tcp_world

    from articulated_fixture import MechanismObserver,array,install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    fixture = args.output/'dishwasher_fixture.usda'
    tab_local = build_fixture(args.usd,fixture)
    mount = (0.,-.39,.59)
    wrist = Rotation.from_euler('y',-45.,degrees=True)*Rotation.from_quat(DOWN_GRIP)
    forward = wrist.apply([0.,0.,1.])
    station.BASES = {'left':(0.,.39,.59),'right':mount}
    parts = {'rack':'/Machine'+RACK,'door':'/Machine'+DOOR,'base':'/Machine'+BASE}
    loads = {'tall_cup':(.529,.120,.654),'short_cup':(.509,-.147,.651)}
    cup_starts = {'tall_cup':(.219,.391,.692),'short_cup':(.312,.479,.686)}
    wood_start = WOOD_START
    rack_sites = RACK_SITES

    def room(*_):
        objects = [scene._box('floor',(4.,4.,.04),(.45,0.,-.02),(.38,.40,.42)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1200.)),
                *[scene._box(side+'_pedestal',(.15,.15,m[2]),(m[0],m[1],m[2]/2),(.20,.23,.26))
                  for side,m in station.BASES.items()]]
        if args.load:
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose
            from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

            for name,point in loads.items():
                objects.append(Object(name=name,object_type=ObjectType.RIGID,
                    usd_path=str((args.load/f'{name}.usda').resolve()),
                    initial_pose=Pose(position_xyz=cup_starts[name] if args.place else point,
                                      rotation_xyzw=(1.,0.,0.,0.)),
                    spawn_cfg_addon={'physics_material':scene._grippy(),'activate_contact_sensors':args.place,
                                     'rigid_props':PhysxRigidBodyPropertiesCfg(
                                         solver_position_iteration_count=16,solver_velocity_iteration_count=6)}))
        if args.place:
            objects.append(scene._box('drainboard',(.201,.205,.04),(.257,.44,.57),(.43,.38,.34)))
            objects.append(scene._box('handwashboard',(.186,.180,.04),(.268,-.446,.57),(.43,.38,.34)))
            objects.append(Object(name='wood_cup',object_type=ObjectType.RIGID,
                usd_path=str((args.load/'wood_cup.usda').resolve()),
                initial_pose=Pose(position_xyz=wood_start,rotation_xyzw=(0.,0.,0.,1.))))
        return objects

    install(fixture,args.output,(0.,0.,0.),(0.,0.,0.,1.),parts,room_factory=room,robot_contact_offset=.001)
    if args.place:
        from isaaclab.sensors import ContactSensorCfg
        from yam_arena.robot import asset_link_paths

        original = scene.build_task_environment
        links = asset_link_paths()

        def build(*p,**kw):
            arena = original(*p,**kw)
            before = arena.env_cfg_callback

            def config(cfg):
                cfg = before(cfg)
                for name in loads:
                    setattr(cfg.scene,'cup_'+name,ContactSensorCfg(
                        prim_path='{ENV_REGEX_NS}/'+name,
                        filter_prim_paths_expr=['{ENV_REGEX_NS}/'+side+'Arm'+links[tip]
                                                for side in ('Left','Right')
                                                for tip in ('tip_left','tip_right')],history_length=0))
                return cfg

            arena.env_cfg_callback = config
            return arena

        scene.build_task_environment = build
    height,width = (176,320) if args.dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True,cam_height=height,cam_width=width,control_hz=20,object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='dishwasher-upper-rack',instruction=INSTRUCTION if args.place else
                              'Pull out the upper rack, then push it back in.')
        env.reset(specification,seed=0)
        base = env._base
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        robot = base.scene['right_robot']
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[-.25,-.91,1.21]],device=base.device),
                                         torch.tensor([[.44,0.,.55]],device=base.device))
        mechanism = MechanismObserver(base.sim.stage)
        action = np.array([*READY,1.,*READY,1.])
        if args.setup_only:
            camera.set_world_poses_from_view(torch.tensor([[-.25,.03,1.52]],device=base.device),
                                             torch.tensor([[.40,.01,.56]],device=base.device))
            for _ in range(5):
                observation = env.step(SimpleNamespace(data=action)).observation
            Image.fromarray(observation.images['top_cam']).save(args.output/'setup.png')
            return {'status':'setup captured','seed':0}
        trace = []
        with (args.output/'trace.jsonl').open('w') as out,record_trial(
                args.output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=RUN_LIMIT:
                    raise RuntimeError('Dishwasher trial reached the action limit')
                observation = env.step(SimpleNamespace(data=action)).observation
                state = mechanism.read()
                joints = next(j for j in state['articulations'].values() if 'Upper_Rack_Assembly_joint' in j['names'])
                extension = joints['positions'][joints['names'].index('Upper_Rack_Assembly_joint')]
                door = np.degrees(joints['positions'][joints['names'].index('Door_joint')])
                forces = {}
                for name in parts:
                    matrix = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    forces[name] = np.linalg.norm(matrix.reshape(-1,6,3).sum(0),axis=1).tolist()
                row = {'step':len(trace),'phase':phase,'action':action.tolist(),'tcp_m':tcp_world(robot).tolist(),
                       'rail_m':float(extension),'door_deg':float(door),'contacts_n':forces,**state}
                if args.load:
                    row['loads'] = {name:array(base.scene.rigid_objects[name].data.root_pose_w)[0].tolist()
                                    for name in loads}
                if args.place:
                    row['wood_pose_xyzw'] = array(base.scene.rigid_objects['wood_cup'].data.root_pose_w)[0].tolist()
                    row['cup_contact_n'] = {
                        name:np.linalg.norm(array(base.scene.sensors['cup_'+name].data.normal_force_matrix_w)
                                             .reshape(-1,4,3).sum(0),axis=1).tolist() for name in loads}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                frame(observation.images['top_cam'])
                if len(trace)==1 or len(trace)%20==0:
                    Image.fromarray(observation.images['top_cam']).save(args.output/'latest.png')
                    print('STEP',len(trace),phase,round(extension,3),round(door,2),flush=True)
                if not args.dream and (sum(forces['door'])>10. or sum(forces['base'])>20.
                                       or sum(forces['rack'])>110.):
                    raise RuntimeError('Dishwasher contact guard exceeded')
                return observation

            if args.dream:
                from dream_control import run_dream

                base.sim.render()
                camera.update(.05,force_recompute=True)
                assets = [args.usd,*[args.load/f'{name}.usda' for name in (*loads,'wood_cup')]]
                return run_dream(args.output,env,specification,env._observation,tick,audit,trace,
                                 action,RUN_LIMIT,assets)

            def hold(steps,phase):
                for _ in range(steps):
                    tick(phase)

            def send(joints,phase):
                action[7:13] = joints
                tick(phase)

            def move(point,seconds,phase,orient=False):
                cartesian_move(lambda:arm_q(robot),send,mount,wrist.as_quat(),point,seconds,phase,
                               interpolate_orientation=orient)

            def place_cup(name):
                arm = base.scene['left_robot']
                base_pos = station.BASES['left']
                cup_wrist = Rotation.from_euler('x',-12.,degrees=True)*Rotation.from_quat(DOWN_GRIP)

                def actuate(q,phase):
                    action[:6] = q
                    tick(phase)

                def go(point,seconds,label,orient=False):
                    cartesian_move(lambda:arm_q(arm),actuate,base_pos,cup_wrist.as_quat(),point,seconds,
                                   name+' '+label,interpolate_orientation=orient)

                initial = np.asarray(trace[-1]['loads'][name])
                grip = initial[:3]+[0.,0.,-.023]-cup_wrist.apply([0.,0.,.010])
                h = .785
                above = np.r_[grip[:2],h]
                action[6] = 1.
                go(above,2.8,'approach',True)
                go(grip,1.7,'lower to cup')
                action[6] = 0.
                hold(24,name+' grasp')
                go(above,2.,'lift')
                carried = np.asarray(trace[-1]['loads'][name])
                if carried[2]-initial[2]<.065 or min(trace[-1]['cup_contact_n'][name][:2])<.10:
                    raise RuntimeError(f'{name} did not lift with bilateral contact')
                rack = np.asarray(trace[-1]['bodies'][parts['rack']]['pose_xyzw'])
                target = rack[:3]+Rotation.from_quat(rack[3:]).apply(rack_sites[name])+[0.,0.,.010]
                setdown = target+tcp_world(arm)-carried[:3]
                hover = np.r_[setdown[:2],h]
                go(hover,3.,'carry to rack')
                go(setdown,2.1,'set down')
                hold(12,name+' seat')
                if abs(trace[-1]['loads'][name][2]-target[2])>.024:
                    raise RuntimeError(f'{name} did not reach the rack surface')
                action[6] = 1.
                hold(25,name+' release')
                go(hover,1.5,'clear rack')

            settle_steps = 230 if args.load and not args.place else 40
            hold(settle_steps,'settle open door and rack')
            closed = trace[-1]
            reset_rail = [row['rail_m'] for row in trace[-20:]]
            if not 89.5<=closed['door_deg']<=90.5 or abs(closed['rail_m'])>.035 or np.ptp(reset_rail)>.001:
                raise RuntimeError('Door or rail failed the reset settling check')
            pose = np.array(closed['bodies'][parts['rack']]['pose_xyzw'])
            handle = pose[:3]+Rotation.from_quat(pose[3:]).apply(tab_local)
            grasp = handle-.011*forward
            action[13] = .45
            move([.29,grasp[1],.71],2.6,'align wrist in front of frame',True)
            move(grasp-.045*forward,1.6,'approach rack clip')
            move(grasp,1.6,'lower over clip')
            action[13] = 0.
            hold(25,'pinch clip')
            if min(trace[-1]['contacts_n']['rack'][4:6])<.10:
                raise RuntimeError('Two fingertips did not grasp the rack clip')
            start = tcp_world(robot).copy()
            move(start+[-.240,0.,0.],4.,'pull upper rack')
            hold(20,'hold rack open')
            opened = trace[-20:]
            if args.place:
                place_cup('tall_cup')
                place_cup('short_cup')
                if min(trace[-1]['contacts_n']['rack'][4:6])<.10:
                    raise RuntimeError('The right gripper did not retain the loaded rack')
            move(start,4.,'return upper rack')
            action[13] = 1.
            hold(20,'release clip')
            move(start-.035*forward,1.6,'retreat from rack')
            hold(25,'verify rack in place')
        tail = trace[-20:]
        travel = min(abs(row['rail_m']-closed['rail_m']) for row in opened)
        error = max(abs(row['rail_m']-closed['rail_m']) for row in tail)
        contact = max(sum(row['contacts_n']['rack']) for row in tail)
        report = {'scope':'source upper rack slide with attached pull clip','steps':len(trace),
                'starting_rail_m':closed['rail_m'],'held_travel_m':travel,
                'final_rail_error_m':error,'final_rack_contact_n':contact,
                'passed':travel>=.225 and error<=.010 and contact<=.1}
        if args.load and not args.place:
            def relative(row,name):
                pose = np.asarray(row['bodies'][parts['rack']]['pose_xyzw'])
                item = np.asarray(row['loads'][name])
                return Rotation.from_quat(pose[3:]).inv().apply(item[:3]-pose[:3])

            report['scope'] = 'two free cups preloaded on the upper rack'
            report['max_load_displacement_m'] = {name:float(max(np.linalg.norm(relative(row,name)-relative(closed,name))
                for row in trace[settle_steps:])) for name in loads}
            report['max_load_tilt_deg'] = {name:float(max(np.rad2deg(np.arccos(np.clip(
                -Rotation.from_quat(row['loads'][name][3:]).as_matrix()[2,2],-1.,1.)))
                for row in trace[settle_steps:])) for name in loads}
            report['passed'] = bool(report['passed'] and max(report['max_load_displacement_m'].values())<=.020
                                     and max(report['max_load_tilt_deg'].values())<=20.)
        if args.place:
            report['scope'] = 'robot loading and return of two upside-down cups'
            report.update(audit(trace))
            report['passed'] = report['complete']
        return report
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--load',type=Path)
    parser.add_argument('--place',action='store_true')
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.dream and args.setup_only:
        parser.error('Choose one of --dream and --setup-only')
    if args.dream or args.setup_only:
        args.place = True
    if args.place and not args.load:
        parser.error('--place requires --load')
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
