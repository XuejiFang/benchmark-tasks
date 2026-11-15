"""Measure aligned and obstructed insertions into a physical USB-C proxy."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np


def add_socket(stage, port, origin, *, extra_port=True, retention=False, strain_support=False):
    from build_key_return_scene import box

    x0,y0,z = origin
    dark = (.20,.23,.26)
    metal = (.60,.64,.67)
    box(stage,port+'/Base',(x0+.043,y0,z-.031),(.083,.105,.044),dark)
    ports = ((0.,'Primary'),(.027,'Secondary')) if extra_port else ((0.,'Primary'),)
    for dy,tag in ports:
        socket = port+'/'+tag
        y = y0+dy
        for sign,label in ((-1,'Minus'),(1,'Plus')):
            box(stage,socket+label+'Wall',(x0+.0054,y+sign*.00585,z),(.0108,.0015,.007),metal)
            box(stage,socket+label+'Lip',(x0+.0054,y,z+sign*.00275),(.0108,.0132,.0012),metal)
        box(stage,socket+'Stop',(x0+.010,y,z),(.002,.014,.008),dark)
        box(stage,socket+'Tongue',(x0+.0071,y,z),(.0045,.0068,.00068),(.11,.14,.17))
    box(stage,port+'/PanelBetween',(x0+.0025,y0+.0134,z),(.005,.0134,.009),dark)
    box(stage,port+'/PanelEnd',(x0+.0025,y0-.0145,z),(.005,.016,.009),dark)
    if strain_support:
        box(stage,port+'/StrainReliefShelf',(x0-.022,y0,z-.0113),(.026,.022,.014),(.32,.34,.36))
    if retention:
        add_retention_spring(stage,port,(x0+.0065,y0,z+.0018))


def add_retention_spring(stage, port, centre):
    """A beveled moving pad supplies connector retention through contact friction."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    from build_key_return_scene import box,rigid
    from build_tea_probe import contact

    anchor,pad = port+'/SpringAnchor',port+'/MovingPad'
    body = rigid(stage,anchor,.01,centre)
    UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
    box(stage,anchor+'/Support',(0.,0.,.012),(.006,.009,.005),(.18,.20,.24))
    rigid(stage,pad,.0008,centre)
    shape = UsdGeom.Mesh.Define(stage,pad+'/Bevel')
    shape.CreatePointsAttr([Gf.Vec3f(x,y,z)
                            for x,z in ((-.002,.0001),(.002,-.00046),(.002,.00065),(-.002,.00065))
                            for y in (-.0024,.0024)])
    shape.CreateFaceVertexCountsAttr([4]*6)
    shape.CreateFaceVertexIndicesAttr([0,2,4,6,1,7,5,3,0,1,3,2,2,3,5,4,4,5,7,6,6,7,1,0])
    shape.CreateSubdivisionSchemeAttr('none')
    shape.CreateDisplayColorAttr([Gf.Vec3f(.77,.62,.37)])
    collider = shape.GetPrim()
    contact(collider,.00016)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('convexHull')
    material = UsdShade.Material.Define(stage,port+'/PadGrip')
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(.92)
    friction.CreateDynamicFrictionAttr(.74)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(material,materialPurpose='physics')
    joint = UsdPhysics.PrismaticJoint.Define(stage,port+'/PadStroke')
    joint.CreateAxisAttr('Z')
    joint.CreateBody0Rel().SetTargets([anchor])
    joint.CreateBody1Rel().SetTargets([pad])
    joint.CreateLowerLimitAttr(-.00035)
    joint.CreateUpperLimitAttr(.0015)
    joint.CreateCollisionEnabledAttr(False)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
    spring.CreateTargetPositionAttr(-.0011)
    spring.CreateStiffnessAttr(1250.)
    spring.CreateDampingAttr(4.)

def add_plug(stage, plug, start, *, gravity=True):
    from pxr import Sdf

    from build_key_return_scene import box,rigid

    metal = (.60,.64,.67)
    body = rigid(stage,plug,.046,start)
    body.AddAppliedSchema('PhysxRigidBodyAPI')
    body.CreateAttribute('physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(not gravity)
    body.CreateAttribute('physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool).Set(True)
    body.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(24)
    blue = (.18,.36,.54)
    box(stage,plug+'/Grip',(-.0192,0.,0.),(.021,.012,.008),blue)
    for sign,label in ((-1,'Minus'),(1,'Plus')):
        box(stage,plug+'/'+label+'Wall',(-.0043,sign*.00405,0.),(.0086,.0007,.0028),metal)
        box(stage,plug+'/'+label+'Skin',(-.0043,0.,sign*.00115),(.0086,.0081,.0005),metal)


def author(path, offset, *, yam=False):
    from pxr import UsdGeom

    from build_key_return_scene import box,stage_at
    from build_tea_probe import contact

    stage = stage_at(path,'Workstation')
    base = '/Workstation'
    add_socket(stage,base+'/Dock',(0.,0.,.053))
    start = (-.16,.07,.053) if yam else (-.016,offset,.053)
    if yam:
        box(stage,base+'/PlugStand',(-.18,.07,.0245),(.053,.042,.049),(.39,.40,.41))
    add_plug(stage,base+'/Plug',start,gravity=yam)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cube):
            contact(prim,.00018)
    stage.GetRootLayer().Save()


def run(args):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera,CameraCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from PIL import Image
    import omni.physics.tensors as tensors
    from pxr import Gf,UsdGeom
    import torch

    from articulated_fixture import array

    asset = args.output/'workstation.usda'
    author(asset,args.offset_mm/1000.)
    dt = 1/480
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt,render_interval=24,device='cpu',use_fabric=False,physics=PhysxCfg(enable_ccd=True)))
    stage = sim.stage
    stage.DefinePrim('/World/Bench','Xform').GetReferences().AddReference(str(asset.resolve()))
    floor = sim_utils.CuboidCfg(size=(.45,.40,.02),collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.42,.37,.30)))
    floor.func('/World/Floor',floor,translation=(0.,0.,-.01))
    dome = sim_utils.DomeLightCfg(intensity=1300.)
    dome.func('/World/Dome',dome)
    cam = UsdGeom.Camera.Define(stage,'/World/Camera')
    cam.CreateClippingRangeAttr(Gf.Vec2f(.005,3.))
    sensor = Camera(CameraCfg(prim_path='/World/Camera',spawn=None,width=640,height=384,data_types=['rgb']))
    sim.reset()
    sensor.set_world_poses_from_view(torch.tensor([[-.105,-.10,.108]],device=sim.device),
                                     torch.tensor([[0.,0.,.051]],device=sim.device))
    physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
    physics.set_subspace_roots('/')
    plug = physics.create_rigid_body_view('/World/Bench/Plug')
    if plug.count!=1:
        raise RuntimeError('Expected one loose connector')
    idx = torch.zeros(1,dtype=torch.int32,device=sim.device)
    trace = []
    with (args.output/'trace.jsonl').open('w') as out:
        for i in range(2400):
            speed = array(plug.get_velocities())[0]
            pose = array(plug.get_transforms())[0]
            active = 240<=i<1920
            force = float(np.clip(13.*(.010-speed[0]),0.,.50)) if active else 0.
            if active:
                input_force = torch.tensor([[force,0.,0.]],device=sim.device)
                plug.apply_forces_and_torques_at_position(input_force,torch.zeros_like(input_force),None,idx,True)
            sim.step(render=False)
            if i%24==23:
                row = {'step':len(trace),'pose_xyzw':pose.tolist(),'force_n':force}
                trace.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                if len(trace)%25==0:
                    sim.render()
                    sensor.update(.05,force_recompute=True)
                    Image.fromarray(array(sensor.data.output['rgb'])[0]).save(args.output/f'frame_{len(trace):03}.png')
                    print('STEP',len(trace),round(float(pose[0])*1000,3),flush=True)
    tail = trace[-10:]
    depth = min(r['pose_xyzw'][0] for r in tail)
    lateral = max(abs(r['pose_xyzw'][1]) for r in tail)
    return {'scope':'USB-C shell and tongue engagement under physical axial force','offset_mm':args.offset_mm,
            'final_min_depth_mm':depth*1000,'final_max_lateral_mm':lateral*1000,
            'seated':depth>=.0075 and lateral<=.001}


def run_yam(output, *, cable=False, full=False, retention=False, strain_support=False, first_only=False,
            raised_collar=False,service_loop=False,dream=False,fixture=None,setup_only=False):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import torch
    from yam_arena import inspect_embodiment,scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q,tcp_world

    from articulated_fixture import MechanismObserver,array,install
    from audit_workstation_trial import INSTRUCTION,audit
    from expert_motion import cartesian_move
    from record_trial import record_trial

    asset = output/'workstation.usda'
    origin = np.array([.390,.220,.75])
    grip_objects = {}
    if cable:
        if fixture is None:
            from run_workstation_cable_probe import author as author_cable

            links,_,_ = author_cable(asset,yam=True,full=full,retention=retention,strain_support=strain_support,
                                      raised_collar=raised_collar,service_loop=service_loop)
            if full:
                collar = '/'+links[1+(len(links)-2)//2]
        else:
            from pxr import Usd

            asset = fixture.resolve()
            stage = Usd.Stage.Open(str(asset))
            root = str(stage.GetDefaultPrim().GetPath())
            collars = [str(p.GetParent().GetPath())[len(root):] for p in stage.Traverse()
                       if p.GetName()=='CableMarker']
            if len(collars)!=1:
                raise ValueError('Workstation USD must define one cable collar')
            collar = collars[0]
        plug = '/Second'
        shift = (0.,0.,.75)
        if full:
            grip_objects = {'first':'/First','collar':collar}
    else:
        author(asset,0.,yam=True)
        plug = '/Plug'
        shift = origin
    grip_objects['plug'] = plug

    def room(*_):
        return [scene._box('worktop',(.60,.72,.04),(.42,.045,.73),(.43,.38,.31)),
                scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.45,.46,.47)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1400.))]

    install(asset,output,shift,(0.,0.,0.,1.),grip_objects,room_factory=room,robot_contact_offset=.0006)
    height,width = (176,320) if dream else (360,640)
    env = inspect_embodiment.YamArenaEmbodiment(headless=True,cam_height=height,cam_width=width,control_hz=20,
        object_jitter=0.,task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='workstation-connector-contact',instruction=INSTRUCTION if full else
                              'Insert the blue connector into the nearer socket.')
        env.reset(specification,seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        cam = base.scene.sensors['top_cam']
        eye,look = ((-.15,-.22,1.27),(.31,0.,.78)) if full else ((.14,-.13,1.12),(.33,.245,.80))
        cam.set_world_poses_from_view(torch.tensor([eye],device=base.device),
                                      torch.tensor([look],device=base.device))
        if setup_only:
            cam.set_world_poses_from_view(torch.tensor([[-.11,-.10,1.37]],device=base.device),
                                          torch.tensor([[.32,0.,.77]],device=base.device))
            base.sim.render()
            cam.update(.05,force_recompute=True)
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        observer = MechanismObserver(base.sim.stage)
        action = np.array([*READY,1.,*READY,1.])
        trace = []
        wrist = Rotation.from_quat(DOWN_GRIP)
        with (output/'trace.jsonl').open('w') as out, record_trial(output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=2400:
                    raise RuntimeError('Workstation trial reached its control limit')
                obs = env.step(SimpleNamespace(data=action)).observation
                states = observer.read()['bodies']
                contacts = {}
                for name in grip_objects:
                    matrix = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    contacts[name] = np.linalg.norm(matrix.reshape(-1,6,3).sum(0),axis=1).tolist()
                pose = states[grip_objects[active]]['pose_xyzw']
                row = {'step':len(trace),'phase':phase,'pose_xyzw':pose,'contacts_n':contacts[active],
                       'action':action.tolist()}
                if cable:
                    row['first_pose_xyzw'] = states['/First']['pose_xyzw']
                if full:
                    row['second_pose_xyzw'] = states['/Second']['pose_xyzw']
                    row['collar_pose_xyzw'] = states[grip_objects['collar']]['pose_xyzw']
                    row['contacts_by_object_n'] = contacts
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                frame(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%20==0:
                    Image.fromarray(obs.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,round((pose[0]-origin[0])*1000,2),flush=True)
                return obs

            def hold(n,phase):
                for _ in range(n):
                    tick(phase)

            def send(q,phase):
                action[arm_joints] = q
                tick(phase)

            def move(point,seconds,phase,orient=False):
                cartesian_move(lambda:arm_q(arm),send,mount,wrist.as_quat(),point,seconds,phase,
                               interpolate_orientation=orient)

            active = 'plug'
            arm = base.scene['left_robot']
            mount = (0.,.305,.75)
            arm_joints = slice(6)
            fingers = slice(1,3)
            gripper = 6
            if dream:
                from dream_control import run_dream

                base.sim.render()
                cam.update(.05,force_recompute=True)
                return run_dream(output,env,specification,env._observation,tick,lambda _:audit(trace),
                                 trace,action,2400,[asset])

            def insert(port):
                nonlocal wrist
                wrist = Rotation.from_quat(DOWN_GRIP)
                hold(30,active+' settle on stand')
                start = np.asarray(trace[-1]['pose_xyzw'][:3])
                grip = start+[-.019,0.,.010]
                move(grip+[0.,0.,.045],2.5,'approach connector',True)
                move(grip,1.6,'lower around grip housing')
                action[gripper] = 0.
                hold(20,'pinch housing')
                move(grip+[0.,0.,.045],1.5,'lift loose connector')
                pose = np.asarray(trace[-1]['pose_xyzw'])
                if pose[2]-start[2]<.030 or min(trace[-1]['contacts_n'][fingers])<.1:
                    raise RuntimeError(f'{active} connector did not lift with both fingertips')
                inverse = Rotation.from_quat(pose[3:]).inv()
                offset = inverse.apply(tcp_world(arm)-pose[:3])
                wrist = inverse*wrist
                standby = port+[-.014,0.,.053]+offset
                move(standby+[0.,0.,.035],2.3,'carry in front of socket',True)
                move(standby,1.8,'align shell at socket mouth')
                command = standby.copy()
                for depth_target in np.linspace(-.012,.0084,14):
                    target = port+[depth_target,0.,.053]
                    for _ in range(5):
                        pose = np.asarray(trace[-1]['pose_xyzw'][:3])
                        correction = target-pose
                        if np.linalg.norm(correction)<=.00045:
                            break
                        command += np.clip(correction*.72,-.0018,.0018)
                        if np.linalg.norm(command-tcp_world(arm))>.011:
                            raise RuntimeError(f'{active} connection stopped progressing at the contact limit')
                        move(command,.26,'align and feed connector')
                        hold(7,'measure connector at socket')
                hold(12,'hold seated plug')
                if trace[-1]['pose_xyzw'][0]-port[0]<.0075:
                    raise RuntimeError(f'{active} connector did not reach the full insertion depth')
                action[gripper] = .65
                hold(18,'release plug')
                move(tcp_world(arm)+[0.,0.,.035],1.5,'withdraw from seated plug')
                hold(20,'settle released plug')

            if full:
                active = 'first'
                arm = base.scene['right_robot']
                mount = (0.,-.305,.75)
                arm_joints = slice(7,13)
                fingers = slice(4,6)
                gripper = 13
                insert(np.array([.390,-.220,.75]))
                if first_only:
                    depth = min(row['first_pose_xyzw'][0]-.390 for row in trace[-20:])
                    force = max(sum(row['contacts_by_object_n']['first']) for row in trace[-20:])
                    return {'scope':'YAM seats the first connector from two loose ends','steps':len(trace),
                            'first_min_depth_mm':depth*1000,'max_final_contact_n':force,
                            'first_seated':depth>=.0075 and force<=.1}
                active = 'plug'
                arm = base.scene['left_robot']
                mount = (0.,.305,.75)
                arm_joints = slice(6)
                fingers = slice(1,3)
                gripper = 6
            insert(origin)
            if full:
                from audit_workstation_trial import GUIDE_X

                active = 'collar'
                hold(10,'settle before routing collar')
                wrist = Rotation.from_quat(DOWN_GRIP)
                collar = np.asarray(trace[-1]['collar_pose_xyzw'])
                if raised_collar:
                    yaw = Rotation.from_quat(collar[3:]).as_euler('xyz')[2]
                    wrist = Rotation.from_rotvec([0.,0.,yaw])*wrist
                pickup = collar[:3]+[0.,0.,.022 if raised_collar else .010]
                action[6] = .60
                move(pickup+[0.,0.,.056],2.0,'approach slack collar',True)
                move(pickup,1.35,'lower to slack collar')
                action[6] = 0.
                hold(16,'grasp slack collar')
                move(pickup+[0.,0.,.022],1.65,'lift slack collar just over guide')
                actual = np.asarray(trace[-1]['collar_pose_xyzw'][:3])
                if actual[2]-collar[2]<.016 or min(trace[-1]['contacts_n'][1:3])<.10:
                    raise RuntimeError('Cable collar failed its bilateral lift')
                place = np.array([GUIDE_X,0.,.761])+tcp_world(arm)-actual
                move(place+[0.,0.,.018],2.25,'carry slack low over guide')
                move(place,1.3,'lower slack to guide')
                action[6] = 1.
                hold(14,'release slack in guide')
                move(place+[0.,0.,.034],1.4,'withdraw from slack guide')
                hold(30,'settle cable and plugs')
        if full:
            return {'scope':'both connectors and slack moved by ordinary YAM commands'} | audit(trace)
        final = np.asarray([r['pose_xyzw'][:3] for r in trace[-20:]])-origin
        contact = max(sum(r['contacts_n']) for r in trace[-20:])
        depth = float(final[:,0].min()*1000)
        lateral = float(np.abs(final[:,1]).max()*1000)
        result = {'scope':'free gravity-enabled connector picked up and inserted by YAM',
                'steps':len(trace),'final_min_depth_mm':depth,'final_max_lateral_mm':lateral,
                'max_final_contact_n':contact,'seated':depth>=7.5 and lateral<=1. and contact<=.1}
        if cable:
            first_depth = float(min(r['first_pose_xyzw'][0]-origin[0] for r in trace)*1000)
            result.update(scope='YAM inserts the second connector of a cable with one seated end',
                          first_min_depth_mm=first_depth,first_stayed_seated=first_depth>=7.5)
        return result
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offset-mm',type=float,default=0.)
    parser.add_argument('--yam',action='store_true')
    parser.add_argument('--cable',action='store_true')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--retention',action='store_true')
    parser.add_argument('--strain-support',action='store_true')
    parser.add_argument('--first-only',action='store_true')
    parser.add_argument('--raised-collar',action='store_true')
    parser.add_argument('--service-loop',action='store_true')
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.dream or args.fixture is not None or args.setup_only:
        args.full = True
    if args.fixture is not None:
        args.raised_collar = True
    if args.full:
        args.cable = True
    if args.retention and not args.cable:
        parser.error('--retention requires a cable trial')
    if (args.strain_support or args.first_only or args.raised_collar or args.service_loop) and not args.full:
        parser.error('Support, collar, service loop and first-only modes require the two-loose-end layout')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,device='cuda:0' if args.yam or args.cable else 'cpu',enable_cameras=True).app
    try:
        if args.yam or args.cable:
            from yam_arena import inspect_embodiment

            inspect_embodiment._APP = app
            report = run_yam(args.output,cable=args.cable,full=args.full,retention=args.retention,
                             strain_support=args.strain_support,first_only=args.first_only,raised_collar=args.raised_collar,
                             service_loop=args.service_loop,dream=args.dream,fixture=args.fixture,setup_only=args.setup_only)
        else:
            report = run(args)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
