"""Stow a real hinged shower seat by contacting its handle and spring keeper."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from run_shower_seat_probe import add_catch
from audit_shower_seat_trial import audit, STEP_LIMIT


ORIGIN = np.array([.51, 0., .035])
FIXTURE_ROTATION = Rotation.from_euler('z', -90., degrees=True)
MOUNTS = {'left': (0., .305, .795), 'right': (0., -.305, .795)}
SEAT_PATH = '/Model/Geometry/Installation/Seat'
CATCH_PATH = '/SeatCatch/Bar'
SEAT_HANDLE = np.array([-.166, -.310, .051])
CATCH_PAD = np.array([0., -.115, -.052])
PIVOT = np.array([0., -.070, .460])
INSTRUCTION = ('Hold the orange retaining lever clear. Fold the shower seat up to the wall, '
               'let the lever click back in front of it, and release the seat.')


def build(source, destination, assist=0.,counterbalance=0.,round_grip=False,handle=SEAT_HANDLE,
          assist_target=86.):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from build_key_return_scene import bar, box, stage_at
    from build_tea_probe import contact

    stage = stage_at(destination, 'Shower')
    root = '/Shower'
    model = stage.DefinePrim(root+'/Model', 'Xform')
    model.GetReferences().AddReference(str(source.resolve()))
    for layer in stage.GetUsedLayers():
        if Path(layer.identifier).name=='animation.usda':
            stage.MuteLayer(layer.identifier)
    for p in list(Usd.PrimRange(model)):
        if p.IsA(UsdPhysics.Scene):
            p.SetActive(False)
    seat = stage.GetPrimAtPath(root+SEAT_PATH)
    seat.AddAppliedSchema('PhysxRigidBodyAPI')
    seat.CreateAttribute('physxRigidBody:disableGravity', Sdf.ValueTypeNames.Bool).Set(False)
    fold = UsdPhysics.RevoluteJoint.Get(stage,root+'/Model/Physics/Joints/fold')
    if not fold:
        raise RuntimeError('The shower-seat hinge is missing')
    if counterbalance:
        mass = UsdPhysics.MassAPI(seat)
        cg = np.array(mass.GetCenterOfMassAttr().Get(),dtype=float)
        pivot = np.array(fold.GetLocalPos1Attr().Get(),dtype=float)
        if not np.all(np.isfinite(cg)):
            cg = np.zeros(3)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*(cg+counterbalance*(pivot-cg))))
    if assist:
        spring = UsdPhysics.DriveAPI.Apply(fold.GetPrim(),'angular')
        spring.CreateTypeAttr('force')
        spring.CreateStiffnessAttr(assist)
        spring.CreateDampingAttr(.025)
        spring.CreateTargetPositionAttr(assist_target)
        spring.CreateMaxForceAttr(10.3)
    inverse = UsdGeom.XformCache().GetLocalToWorldTransform(seat).GetInverse()
    home = PIVOT+handle

    def seat_local(vector):
        return inverse.Transform(Gf.Vec3d(*(home+vector)))

    grip_path = root+SEAT_PATH+'/GrabRail'
    if round_grip:
        parts = [bar(stage,grip_path,seat_local([-.026,0.,0.]),seat_local([.026,0.,0.]),
                     .016,(.22,.38,.43))]
    else:
        box(stage,grip_path,tuple(seat_local([0.,0.,0.])),(.062,.018,.023),(.22,.38,.43))
        box(stage,grip_path+'Lip',tuple(seat_local([0.,0.,.022])),(.060,.036,.008),(.22,.38,.43))
        box(stage,grip_path+'Web',tuple(seat_local([0.,0.,.013])),(.060,.016,.014),(.22,.38,.43))
        parts = [stage.GetPrimAtPath(grip_path+ending) for ending in ('','Lip','Web')]
    for label, dx in (('L',-.029),('R',.029)):
        bar(stage,root+SEAT_PATH+'/GrabRailFoot'+label,seat_local([dx,0.,-.040]),
            seat_local([dx,0.,0.]),.0045,(.23,.31,.35))
    extension = .059+max(0.,abs(handle[0])-.166)
    bar(stage,root+SEAT_PATH+'/SideGrabMount',seat_local([extension,0.,-.041]),
        seat_local([-.029,0.,-.041]),.007,(.25,.30,.32))
    catch = add_catch(stage,root+'/SeatCatch',stiffness=.001)
    anchor = UsdPhysics.RevoluteJoint.Get(stage,root+'/SeatCatch/Keeper')
    local_pivot = np.asarray(anchor.GetLocalPos0Attr().Get())
    anchor.GetLocalPos0Attr().Set(Gf.Vec3f(*(ORIGIN+FIXTURE_ROTATION.apply(local_pivot))))
    qx,qy,qz,qw = FIXTURE_ROTATION.as_quat()
    anchor.CreateLocalRot0Attr(Gf.Quatf(qw,qx,qy,qz))
    bar(stage,catch+'/RockerShaft',(0.,0.,0.),(0.,-.115,0.),.006,(.41,.39,.34))
    knob = bar(stage,catch+'/RockerPad',(0.,-.115,0.),tuple(CATCH_PAD),.009,(.89,.49,.15))
    grip = UsdShade.Material.Define(stage,root+'/GripMaterial')
    surface = UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
    surface.CreateStaticFrictionAttr(1.20)
    surface.CreateDynamicFrictionAttr(.95)
    for part in [*parts,knob]:
        contact(part,.00045)
        UsdShade.MaterialBindingAPI.Apply(part).Bind(grip,materialPurpose='physics')
    stage.GetRootLayer().Save()


def run(source, output, assist=0.,wrist_follow=0.,counterbalance=0.,round_grip=False,side_grip=.166,
        lead_limit=17.,arm_diagnostics=False,assist_target=86.,radial_grip=.310,left_offset=0.,
        drive_test=False,entry_pitch=0.,left_y=None,left_height=.795,release_test=False,dream=False,
        setup_only=False):
    import isaaclab.sim as sim_utils
    from inspect_robots import Scene
    from PIL import Image
    import torch
    from yam_arena import ik, inspect_embodiment, scene, station
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import MechanismObserver, array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    asset = output/'shower_seat.usda'
    local_handle = SEAT_HANDLE.copy()
    local_handle[0] = -side_grip
    local_handle[1] = -radial_grip
    build(source,asset,assist,counterbalance,round_grip,local_handle,assist_target)
    shoulder_y = .305+max(0.,side_grip-.166)*.36 if left_y is None else left_y
    mounts = MOUNTS | {'left':(left_offset,shoulder_y,left_height)}
    station.BASES = mounts.copy()
    bodies = {'seat':SEAT_PATH, 'catch':CATCH_PATH}

    def room(*_):
        return [scene._box('floor',(3.,3.,.04),(.40,0.,-.02),(.44,.44,.42)),
                scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1510.))]

    install(asset,output,ORIGIN,FIXTURE_ROTATION.as_quat(),bodies,
            room_factory=room,robot_contact_offset=.00065,track_friction=arm_diagnostics)
    if arm_diagnostics:
        from isaaclab.sensors import ContactSensorCfg
        from yam_arena.robot import asset_link_paths

        link_paths = asset_link_paths()
        links = tuple(link_paths)
        old_build = scene.build_task_environment

        def add_arm_contacts(*args,**kwargs):
            arena = old_build(*args,**kwargs)
            old_callback = arena.env_cfg_callback

            def configure(cfg):
                cfg = old_callback(cfg)
                cfg.scene.seat_arm_contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/mechanism'+SEAT_PATH,
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/LeftArm'+link_paths[name] for name in links],
                    history_length=0,max_contact_data_count_per_prim=320)
                return cfg

            arena.env_cfg_callback = configure
            return arena

        scene.build_task_environment = add_arm_contacts
    env = inspect_embodiment.YamArenaEmbodiment(
        headless=True,cam_height=360,cam_width=640,control_hz=20,object_jitter=0.,
        task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        env.reset(Scene(id='fold-down-shower-seat',instruction=INSTRUCTION),seed=0)
        env._task_step = lambda: SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        camera.set_world_poses_from_view(torch.tensor([[.02,-.86,1.35]],device=base.device),
                                         torch.tensor([[.38,.01,.73]],device=base.device))
        if setup_only:
            base.sim.render()
            camera.update(.05,force_recompute=True)
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        observer = MechanismObserver(base.sim.stage)
        action = np.array([*READY,1.,*READY,1.])
        down = Rotation.from_quat(DOWN_GRIP)
        left_home = (Rotation.from_euler('y',entry_pitch,degrees=True)
                     *Rotation.from_euler('z',-90.,degrees=True)*down)
        wrists = {'left':left_home,'right':down}
        pivot = ORIGIN+FIXTURE_ROTATION.apply(PIVOT)
        handle = FIXTURE_ROTATION.apply(local_handle)
        trace = []

        with (output/'trace.jsonl').open('w') as raw, record_trial(output,True) as record:
            def step(phase,inference_latency=None):
                obs = env.step(SimpleNamespace(data=action)).observation
                state = observer.read()
                poses = {name:state['bodies'][path]['pose_xyzw'] for name,path in bodies.items()}
                joint = next(j for j in state['articulations'].values() if 'fold' in j['names'])
                seat = float(np.rad2deg(joint['positions'][joint['names'].index('fold')]))
                relative = FIXTURE_ROTATION.inv()*Rotation.from_quat(poses['catch'][3:])
                catch = float(np.rad2deg(relative.as_rotvec()[1]))
                contacts = {}
                for name in bodies:
                    force = array(base.scene.sensors['mechanism_contact_'+name].data.normal_force_matrix_w)
                    contacts[name] = np.linalg.norm(force.reshape(-1,6,3).sum(0),axis=1).tolist()
                row = {'step':len(trace),'phase':phase,'action':action.tolist(),'fold_deg':seat,
                       'catch_deg':catch,'poses':poses,'contacts_n':contacts,
                       'tcp':{a:tcp_world(base.scene[a+'_robot']).tolist() for a in MOUNTS}}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                if arm_diagnostics:
                    forces = array(base.scene.sensors['seat_arm_contacts'].data.normal_force_matrix_w)
                    row['left_link_contacts_n'] = dict(zip(links,np.linalg.norm(
                        forces.reshape(-1,len(links),3).sum(0),axis=1).tolist()))
                    row['left_joint_error_rad'] = (arm_q(base.scene['left_robot'])-action[:6]).tolist()
                    grip_data = base.scene.sensors['mechanism_contact_seat'].data
                    normals = array(grip_data.normal_force_matrix_w).reshape(-1,6,3)
                    friction = array(grip_data.friction_force_matrix_w).reshape(-1,6,3)
                    grip_force = (normals+friction)[:,:3].sum(axis=(0,1))
                    lever = Rotation.from_euler('y',seat,degrees=True).apply(handle)
                    row['left_grip_force_world_n'] = grip_force.tolist()
                    row['estimated_hinge_torque_nm'] = float(np.cross(lever,grip_force)[1])
                trace.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
                raw.flush()
                record(obs.images['top_cam'])
                if len(trace)==1 or len(trace)%25==0:
                    Image.fromarray(obs.images['top_cam']).save(output/'latest.png')
                    print('STEP',len(trace),phase,round(seat,2),round(catch,2),flush=True)
                if any(sum(f)>180. for f in contacts.values()):
                    raise RuntimeError('Shower seat contact exceeded 180 N')
                return obs

            def hold(n,phase):
                for _ in range(n):
                    step(phase)

            def move(side, position, seconds, phase, orient=False):
                index = 0 if side=='left' else 7
                arm = base.scene[side+'_robot']

                def send(q, phase):
                    action[index:index+6] = q
                    step(phase)

                cartesian_move(lambda:arm_q(arm),send,mounts[side],wrists[side].as_quat(),position,
                               seconds,phase,interpolate_orientation=orient)

            def grip(side,name,target):
                index = 0 if side=='left' else 7
                fingertip = 1 if side=='left' else 4
                clear = target + ([0.,0.,.057] if side=='left' else [0.,-.012,.047])
                action[index+6] = .79 if side=='left' else .46
                if side=='left' and entry_pitch:
                    clear = target-wrists[side].apply([0.,0.,.040])
                    current = arm_q(base.scene['left_robot']).copy()
                    aligned, distance, angle = ik.solve(clear,wrists[side].as_quat(),
                                                        current,mounts[side],iters=250)
                    if distance>.001 or angle>np.deg2rad(.3):
                        raise RuntimeError('Angled seat approach is out of reach')
                    for fraction in np.linspace(0.,1.,61)[1:]:
                        action[:6] = current + ik.smoothstep(fraction)*(aligned-current)
                        step('approach the side of the seat rail')
                else:
                    move(side,clear,2.55,'approach '+name,True)
                move(side,target,1.5,'enter '+name+' grip')
                hold(12,'settle at '+name)
                action[index+6] = 0.
                hold(23,'grasp '+name)
                if min(trace[-1]['contacts_n'][name][fingertip:fingertip+2])<=.1:
                    raise RuntimeError(name+' did not make bilateral contact')

            if dream:
                from dream_control import run_dream
                return run_dream(output,env,Scene(id='fold-down-shower-seat',instruction=INSTRUCTION),
                                 env._observation,step,audit,trace,action,STEP_LIMIT,[asset],
                                 early_stop_interval=20)

            hold(32,'settle physical seat and catch')
            catch_pose = np.asarray(trace[-1]['poses']['catch'])
            resting = catch_pose[:3]+Rotation.from_quat(catch_pose[3:]).apply(CATCH_PAD)
            action[13] = 0.
            move('right',resting+[0.,0.,.059],2.65,'approach the orange rocker',True)
            move('right',resting+[0.,0.,.010],1.25,'touch the orange rocker')
            bias_x = 0.
            for _ in range(24):
                current = trace[-1]['catch_deg']
                if current>=-16.:
                    break
                catch_pose = np.asarray(trace[-1]['poses']['catch'])
                pad = Rotation.from_quat(catch_pose[3:]).apply(CATCH_PAD)
                following = Rotation.from_euler('x',8.,degrees=True).apply(pad)
                pressure = sum(trace[-1]['contacts_n']['catch'][3:])
                advance = .013 if pressure<6. else .006
                error_x = catch_pose[0]+following[0]-trace[-1]['tcp']['right'][0]
                bias_x = float(np.clip(bias_x+.6*error_x,-.040,.040))
                move('right',catch_pose[:3]+following+[bias_x,0.,-advance],.3,'press rocker to open keeper')
                hold(5,'measure keeper clearance')
            hold(12,'hold catch open')
            if trace[-1]['catch_deg'] < -18.:
                raise RuntimeError('Keeper did not clear the folding seat')
            if drive_test:
                joint = next(view for view in observer.joints.values()
                             if view.count and 'fold' in view.shared_metatype.dof_names)
                index = joint.shared_metatype.dof_names.index('fold')
                ids = torch.zeros(1,dtype=torch.int32,device=base.device)
                torques = torch.zeros((1,len(joint.shared_metatype.dof_names)),device=base.device)
                for _ in range(160):
                    angle = trace[-1]['fold_deg']
                    if angle>=89.:
                        break
                    torques[0,index] = min(16.,4.+max(0.,88.-angle)*.21)
                    joint.set_dof_actuation_forces(torques,ids)
                    step('test the hinge under a measured load')
                torques.zero_()
                joint.set_dof_actuation_forces(torques,ids)
                return {'scope':'fixture limit diagnostic with direct hinge actuation',
                        'maximum_fold_deg':max(r['fold_deg'] for r in trace),
                        'keeper_deg':trace[-1]['catch_deg']}
            grip('left','seat',pivot+handle+[0.,0.,.003])
            seat_offset = np.asarray(trace[-1]['tcp']['left'])-pivot-handle
            best = trace[-1]['fold_deg']
            stalled = 0
            lead = 3.
            for _ in range(112):
                current = trace[-1]['fold_deg']
                if current>=89. or stalled>=max(18,round(lead_limit)):
                    break
                command = min(90.,current+lead)
                rotation = Rotation.from_euler('y',command,degrees=True)
                palm_angle = wrist_follow*command
                if wrist_follow>0. and not entry_pitch:
                    palm_angle *= float(np.clip((85.-command)/23.,0.,1.))
                wrist_adjustment = Rotation.from_euler('y',palm_angle,degrees=True)
                wrists['left'] = wrist_adjustment*left_home
                move('left',pivot+rotation.apply(handle)+wrist_adjustment.apply(seat_offset),.37,
                     'fold hinged seat',True)
                observed = trace[-1]['fold_deg']
                lead = float(np.clip(lead+.55*(2.-(observed-current)),2.2,lead_limit))
                if observed>best+.30:
                    best = observed
                    stalled = 0
                else:
                    stalled += 1
            hold(18,'hold seat against the wall')
            if trace[-1]['fold_deg']<86.:
                raise RuntimeError('Seat did not reach the stowed angle')
            right = np.asarray(trace[-1]['tcp']['right'])
            move('right',right+[0.,-.035,.043],1.2,'withdraw from keeper')
            action[13] = 1.
            hold(14,'let keeper engage')
            hold(12,'verify keeper before releasing seat')
            action[6] = 1.
            hold(15,'release seat')
            left = np.asarray(trace[-1]['tcp']['left'])
            move('left',left+[-.035,0.,.055],1.2,'withdraw from seat')
            for side,index in (('left',0),('right',7)):
                initial = arm_q(base.scene[side+'_robot']).copy()
                for fraction in np.linspace(0.,1.,51)[1:]:
                    action[index:index+6] = initial+ik.smoothstep(fraction)*(np.asarray(READY)-initial)
                    step('park the '+side+' arm away from the fixture')
            hold(27,'observe stowed seat after release')
            if release_test:
                index = torch.zeros(1,dtype=torch.int32,device=base.device)
                force = torch.zeros((1,3),device=base.device)
                seat_body = observer.bodies[SEAT_PATH]
                before_load = trace[-1]['fold_deg']
                for _ in range(30):
                    seat_body.apply_forces_and_torques_at_position(
                        force,torch.tensor([[0.,-.60,0.]],device=base.device),None,index,True)
                    step('diagnostic downward load on the released seat')
                loaded = min(row['fold_deg'] for row in trace[-30:])
                catch_pose = np.asarray(trace[-1]['poses']['catch'])
                pad = catch_pose[:3]+Rotation.from_quat(catch_pose[3:]).apply(CATCH_PAD)
                action[13] = 0.
                move('right',pad+[0.,0.,.059],2.65,'approach keeper for diagnostic release',True)
                move('right',pad+[0.,0.,.010],1.25,'touch keeper for diagnostic release')
                bias_x = 0.
                for _ in range(24):
                    if trace[-1]['catch_deg']>=-16.:
                        break
                    catch_pose = np.asarray(trace[-1]['poses']['catch'])
                    pad = Rotation.from_quat(catch_pose[3:]).apply(CATCH_PAD)
                    following = Rotation.from_euler('x',8.,degrees=True).apply(pad)
                    pressure = sum(trace[-1]['contacts_n']['catch'][3:])
                    advance = .013 if pressure<6. else .006
                    error_x = catch_pose[0]+following[0]-trace[-1]['tcp']['right'][0]
                    bias_x = float(np.clip(bias_x+.6*error_x,-.040,.040))
                    move('right',catch_pose[:3]+following+[bias_x,0.,-advance],.3,
                         'press keeper to test retention')
                    hold(5,'observe released keeper')
                hold(45,'observe the seat after opening the keeper')
                return {'scope':'diagnostic external loading after robot withdrawal',
                        'before_load_deg':before_load,'minimum_loaded_deg':loaded,
                        'after_keeper_release_deg':trace[-1]['fold_deg'],
                        'after_keeper_release_catch_deg':trace[-1]['catch_deg']}
        final = trace[-20:]
        minimum = min(r['fold_deg'] for r in final)
        keeper = max(r['catch_deg'] for r in final)
        touch = max(sum(sum(f) for f in r['contacts_n'].values()) for r in final)
        initially_open = max(abs(row['fold_deg']) for row in trace[:32])<=3.
        return {'controls':len(trace),'fold_deg_min':minimum,'catch_deg_max':keeper,'contact_n_max':touch,
                'hinge_spring_nm_per_degree':assist,'rests_open':initially_open,
                'complete':initially_open and audit(trace)['score']==1.} | audit(trace)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--assist',type=float,default=0.,help='Fixed passive hinge stiffness in N m/degree')
    parser.add_argument('--assist-target',type=float,default=86.)
    parser.add_argument('--wrist-follow',type=float,default=0.,help='Fraction of the hinge angle followed by the wrist')
    parser.add_argument('--counterbalance',type=float,default=0.)
    parser.add_argument('--round-grip',action='store_true')
    parser.add_argument('--side-grip',type=float,default=.166)
    parser.add_argument('--radial-grip',type=float,default=.310)
    parser.add_argument('--left-offset',type=float,default=0.)
    parser.add_argument('--left-y',type=float)
    parser.add_argument('--left-height',type=float,default=.795)
    parser.add_argument('--entry-pitch',type=float,default=0.)
    parser.add_argument('--lead-limit',type=float,default=17.)
    parser.add_argument('--arm-diagnostics',action='store_true')
    parser.add_argument('--drive-test',action='store_true')
    parser.add_argument('--release-test',action='store_true')
    parser.add_argument('--dream',action='store_true')
    parser.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if not 0.<=args.counterbalance<1.:
        parser.error('--counterbalance must be at least zero and less than one')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True).app
    try:
        from yam_arena import inspect_embodiment
        inspect_embodiment._APP = app
        result = run(args.source,args.output,args.assist,args.wrist_follow,args.counterbalance,args.round_grip,
                     args.side_grip,args.lead_limit,args.arm_diagnostics,args.assist_target,args.radial_grip,
                     args.left_offset,args.drive_test,args.entry_pitch,args.left_y,args.left_height,args.release_test,
                     args.dream,args.setup_only)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
