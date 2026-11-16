"""Try opening and closing a physical folding board with a native PhysX towel."""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from audit_towel_trial import CUBBY_CENTER, CUBBY_SIZE, INSTRUCTION, STEP_LIMIT, audit

PIVOT = np.array([.31,0.,.735])
TAB = np.array([.174,-.155,.025])
OPEN_TAB = np.array([.174,-.174,-.028])
MOUNT = (0.,-.305,.75)


def author(destination,pickup=False):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    from build_key_return_scene import bar, box, rigid, stage_at
    from build_tea_probe import contact

    stage = stage_at(destination,'FoldingBoard')
    root = '/FoldingBoard'
    fixed_width = .180 if pickup else .291
    box(stage,root+'/FixedHalf',(-.099,0.,-.005),(.191,fixed_width,.010),(.21,.38,.39))
    if pickup:
        box(stage,root+'/Riser',(-.10,0.,-.0175),(.17,.155,.015),(.19,.28,.28))
    flap = root+'/Flap'
    rigid(stage,flap,.280,(0.,0.,0.))
    box(stage,flap+'/Panel',(.098,0.,-.005),(.190,.291,.010),(.22,.44,.44))
    bar(stage,flap+'/GripStem',(.174,-.14,-.002),tuple(TAB),.0045,(.29,.48,.35))
    knob = UsdGeom.Sphere.Define(stage,flap+'/GripKnob')
    knob.CreateRadiusAttr(.013)
    knob.AddTranslateOp().Set(Gf.Vec3d(*TAB))
    knob.CreateDisplayColorAttr([Gf.Vec3f(.51,.64,.43)])
    bar(stage,flap+'/OpenStem',(.174,-.141,-.009),tuple(OPEN_TAB),.0045,(.36,.29,.25))
    opener = UsdGeom.Sphere.Define(stage,flap+'/OpenKnob')
    opener.CreateRadiusAttr(.013)
    opener.AddTranslateOp().Set(Gf.Vec3d(*OPEN_TAB))
    opener.CreateDisplayColorAttr([Gf.Vec3f(.68,.36,.25)])
    hinge = UsdPhysics.RevoluteJoint.Define(stage,root+'/Hinge')
    hinge.CreateBody1Rel().SetTargets([flap])
    hinge.CreateAxisAttr('Y')
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*PIVOT))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.))
    hinge.CreateLowerLimitAttr(-173.)
    hinge.CreateUpperLimitAttr(0.)
    damper = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular')
    damper.CreateDampingAttr(.009)
    damper.CreateStiffnessAttr(0.)
    lining = UsdShade.Material.Define(stage,root+'/Lining')
    material = UsdPhysics.MaterialAPI.Apply(lining.GetPrim())
    material.CreateStaticFrictionAttr(.95)
    material.CreateDynamicFrictionAttr(.84)
    for path in (root+'/FixedHalf',flap+'/Panel',flap+'/GripKnob',flap+'/OpenKnob'):
        part = stage.GetPrimAtPath(path)
        contact(part,.0012)
        UsdShade.MaterialBindingAPI.Apply(part).Bind(lining,materialPurpose='physics')
    stage.GetRootLayer().Save()


def run(output,pickup=False,cubby=False,asset_path=None,dream=False,setup_only=False):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg
    from inspect_robots import Scene
    from PIL import Image
    import omni.physics.tensors as tensors
    import torch
    from yam_arena import ik, inspect_embodiment, scene
    from yam_arena.constants import READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.runtime import arm_q, tcp_world
    from articulated_fixture import array, install
    from expert_motion import cartesian_move
    from record_trial import record_trial

    asset = asset_path or output/'folding_board.usda'
    if asset_path is None:
        author(asset,pickup)

    def room(*_):
        table_x = .51 if cubby else .63
        surfaces = [scene._box('worktop',(table_x,.286,.04),(.645-table_x/2,0.,.690 if pickup else .702),(.43,.35,.27)),
                    scene._box('floor',(3.,3.,.04),(.45,0.,-.02),(.40,.41,.40)),
                    scene.DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=1530.))]
        if cubby:
            wood = (.48,.37,.27)
            surfaces.extend([
                scene._box('shelf',tuple(CUBBY_SIZE),tuple(CUBBY_CENTER),wood),
                scene._box('shelf_back',(.012,.268,.014),tuple(CUBBY_CENTER+[.098,0.,.014]),wood),
                scene._box('shelf_left',(.205,.009,.014),tuple(CUBBY_CENTER+[0.,.130,.014]),wood),
                scene._box('shelf_right',(.205,.009,.014),tuple(CUBBY_CENTER+[0.,-.130,.014]),wood),
                scene._box('shelf_support',(.033,.216,.128),tuple(CUBBY_CENTER+[-.086,0.,-.071]),wood),
            ])
        return surfaces

    install(asset,output,PIVOT,(0.,0.,0.,1.),{'flap':'/Flap'},room_factory=room,robot_contact_offset=.0008)
    default_room = scene.build_task_environment

    def configure(*a,**kw):
        arena = default_room(*a,**kw)
        callback = arena.env_cfg_callback

        def scene_cfg(cfg):
            cfg = callback(cfg)
            cfg.sim.dt = 1/240
            cfg.decimation = cfg.sim.render_interval = 12
            cfg.sim.physics = PhysxCfg()
            cfg.scene.towel = DeformableObjectCfg(
                prim_path='{ENV_REGEX_NS}/Towel',
                init_state=DeformableObjectCfg.InitialStateCfg(pos=tuple(PIVOT+[0.,0.,.012])),
                spawn=sim_utils.MeshRectangleCfg(size=(.36,.24),edge_refinement=12,
                    deformable_props=PhysxDeformableBodyPropertiesCfg(),
                    collision_props=[PhysxCollisionCfg(rest_offset=.00065,contact_offset=.0035)],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.72,.43,.24)),
                    physics_material=PhysxSurfaceDeformableBodyMaterialCfg(density=380.,surface_thickness=.002,
                        poissons_ratio=.27,youngs_modulus=4.5e5,surface_bend_stiffness=1.3e4,
                        elasticity_damping=.16,bend_damping=.11,static_friction=.91,dynamic_friction=.80)))
            return cfg

        arena.env_cfg_callback = scene_cfg
        return arena

    scene.build_task_environment = configure
    width,height = (320,176) if dream else (640,360)
    env = inspect_embodiment.YamArenaEmbodiment(device='cuda:0',headless=True,cam_width=width,cam_height=height,
        control_hz=20,object_jitter=0.,task=str(Path(__file__).with_name('press_fixture_task.py')))
    try:
        specification = Scene(id='folding-board-contact',
            instruction=INSTRUCTION if cubby else 'Fold the towel using the folding board.')
        env.reset(specification,seed=0)
        env._task_step = lambda:SimpleNamespace(observation=env._observation())
        base = env._base
        camera = base.scene.sensors['top_cam']
        eye = ([.68,0.,1.25] if setup_only else [-.16,-.52,1.23]) if cubby else [.08,-.49,1.18]
        focus = [.22,0.,.71] if cubby else [.31,0.,.75]
        camera.set_world_poses_from_view(torch.tensor([eye],device=base.device),
                                         torch.tensor([focus],device=base.device))
        if dream or setup_only:
            base.sim.render()
            camera.update(.05,force_recompute=True)
        if setup_only:
            Image.fromarray(env._observation().images['top_cam']).save(output/'setup.png')
            return {'scope':'fixed reset screenshot','seed':0}
        cloth = base.scene['towel']
        initial = array(cloth.data.nodal_pos_w)[0]
        edges = [initial[:,0]>PIVOT[0]+.105,initial[:,0]<PIVOT[0]-.105]
        physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
        physics.set_subspace_roots('/')
        flap = physics.create_rigid_body_view('/World/envs/env_0/mechanism/Flap')
        if flap.count!=1 or min(edge.sum() for edge in edges)<4:
            raise RuntimeError('The flap and two cloth edges must be present')
        arm = base.scene['right_robot']
        left = base.scene['left_robot']
        action = np.array([*READY,1.,*READY,1.])
        wrist = Rotation.from_euler('y',-12.,degrees=True)*Rotation.from_quat(DOWN_GRIP)
        trace = []

        with (output/'trace.jsonl').open('w') as log,record_trial(output,True,width=width,height=height) as frame:
            def tick(phase,inference_latency=None):
                if len(trace)>=STEP_LIMIT:
                    raise RuntimeError('Towel trial exceeded its control limit')
                obs = env.step(SimpleNamespace(data=action)).observation
                pose = array(flap.get_transforms())[0]
                rot = Rotation.from_quat(pose[3:])
                axis = rot.apply([1.,0.,0.])
                board_deg = float(np.degrees(np.arctan2(axis[2],axis[0])))
                points = array(cloth.data.nodal_pos_w)[0]
                values = array(base.scene.sensors['mechanism_contact_flap'].data.normal_force_matrix_w)
                forces = np.linalg.norm(values.reshape(-1,6,3).sum(0),axis=1)
                row = {'step':len(trace),'phase':phase,'board_deg':board_deg,'flap_pose_xyzw':pose.tolist(),
                       'towel_width_m':float(np.ptp(points[:,0])),'edge_x_m':[float(np.median(points[e,0])) for e in edges],
                       'cloth_z_p10_m':float(np.quantile(points[:,2],.10)),
                       'cloth_min_xyz':points.min(0).tolist(),'cloth_max_xyz':points.max(0).tolist(),
                       'left_tcp_xyz_m':tcp_world(left).tolist(),
                       'force_n':forces.tolist(),'tcp_xyz_m':tcp_world(arm).tolist(),'action':action.tolist()}
                if inference_latency is not None:
                    row['inference_latency_s'] = inference_latency
                trace.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                frame(obs.images['top_cam'])
                if len(trace)%25==0:
                    print('STEP',len(trace),phase,round(board_deg,2),round(row['towel_width_m'],3),flush=True)
                return obs

            if dream:
                from dream_control import run_dream

                return run_dream(output,env,specification,env._observation,tick,lambda _:audit(trace),
                                 trace,action,STEP_LIMIT,[asset])

            def hold(count,phase):
                for _ in range(count):
                    tick(phase)

            def send(q,phase):
                action[7:13] = q
                tick(phase)

            def move(point,seconds,phase,orient=False):
                cartesian_move(lambda:arm_q(arm),send,MOUNT,wrist.as_quat(),point,seconds,phase,
                               interpolate_orientation=orient)

            hold(37,'settle the unfolded towel')
            approach = PIVOT+TAB-wrist.apply([0.,0.,.005])
            q = np.copy(arm_q(arm))
            target,pe,re = ik.solve(approach+[0.,-.018,.055],wrist.as_quat(),q,MOUNT,iters=270)
            if pe>.001 or re>np.deg2rad(.3):
                raise RuntimeError(f'The board handle is outside the wrist envelope: {pe:.4f}, {re:.4f}')
            action[13] = .70
            for alpha in np.linspace(0.,1.,52)[1:]:
                send(q+ik.smoothstep(alpha)*(target-q),'approach the board')
            move(approach,1.65,'enter the board-handle grasp')
            action[13] = 0.
            hold(21,'grip the board handle')
            if min(trace[-1]['force_n'][4:])<=.05:
                raise RuntimeError('The board knob did not reach both fingers')
            offset = tcp_world(arm)-PIVOT-TAB
            for angle in np.linspace(3.,171.,36):
                rotation = Rotation.from_euler('y',-angle,degrees=True)
                move(PIVOT+rotation.apply(TAB)+offset,.29,'fold the towel with the board')
            hold(27,'let the folded towel settle')
            action[13] = .72
            hold(13,'release the folded board before changing the grasp')
            move(tcp_world(arm)+[0.,-.023,.052],1.4,'clear the folded board')
            pose = np.asarray(trace[-1]['flap_pose_xyzw'])
            knob = pose[:3]+Rotation.from_quat(pose[3:]).apply(OPEN_TAB)
            grasp = knob-wrist.apply([0.,0.,.005])
            move(grasp+[0.,0.,.043],1.3,'approach the knob on the folded side')
            move(grasp,1.4,'lower over the knob on the folded side')
            action[13] = 0.
            hold(20,'grip the knob to open the board')
            if min(trace[-1]['force_n'][4:])<=.05:
                raise RuntimeError('The raised board knob did not reach both fingers')
            offset = tcp_world(arm)-knob
            start_angle = trace[-1]['board_deg']
            for angle in np.linspace(start_angle-3.,0.,35):
                rotation = Rotation.from_euler('y',-angle,degrees=True)
                move(PIVOT+rotation.apply(OPEN_TAB)+offset,.29,'return the board to the table')
            action[13] = .78
            hold(17,'release the returned board')
            move(tcp_world(arm)+[0.,-.020,.052],1.6,'withdraw the right arm')
            hold(25,'observe the folded towel')
            fold_tail = trace[-20:]
            if pickup:
                points = array(cloth.data.nodal_pos_w)[0]
                if max(r['towel_width_m'] for r in fold_tail)>.210:
                    raise RuntimeError('The towel spread too far after the board was reopened')
                arms = (left,arm)
                mounts = ((0.,.305,.75),MOUNT)
                angles = [Rotation.from_euler('z',-(15. if cubby else 10.)*s,degrees=True)*
                          Rotation.from_euler('x',-80.*s,degrees=True)*Rotation.from_quat(DOWN_GRIP)
                          for s in (1.,-1.)]
                grasps = []
                for s in (1.,-1.):
                    rim = points[(s*points[:,1]>.096)&(np.abs(points[:,0]-.206)<.035)]
                    if len(rim)<4:
                        raise RuntimeError('Cannot locate both exposed towel rims')
                    grasps.append(np.array([.206,s*(np.max(s*rim[:,1])-.011),np.median(rim[:,2])+.001]))

                def approach_both(targets,seconds,phase):
                    configs = [np.copy(arm_q(robot)) for robot in arms]
                    desired = []
                    for target,orientation,q,mount in zip(targets,angles,configs,mounts):
                        next_q,pe,re = ik.solve(target,orientation.as_quat(),q,mount,iters=240)
                        if pe>.001 or re>np.deg2rad(.3):
                            raise RuntimeError(f'{phase}: {pe:.4f} m, {re:.4f} rad')
                        desired.append(next_q)
                    for fraction in np.linspace(0.,1.,round(seconds*20)+1)[1:]:
                        for side,(start,end) in enumerate(zip(configs,desired)):
                            action[7*side:7*side+6] = start+ik.smoothstep(fraction)*(end-start)
                        tick(phase)

                def translate_both(targets,seconds,phase,end_angles=None):
                    configs = [np.copy(arm_q(robot)) for robot in arms]
                    starts = [ik.tcp_pose(q,mount)[0] for q,mount in zip(configs,mounts)]
                    for fraction in np.linspace(0.,1.,round(seconds*20)+1)[1:]:
                        for side,(start,target,orientation,mount) in enumerate(zip(starts,targets,angles,mounts)):
                            point = start+ik.smoothstep(fraction)*(target-start)
                            quaternion = orientation.as_quat()
                            if end_angles:
                                target_rotation = end_angles[side].as_matrix()
                                quaternion = ik.matrix_to_quat_xyzw(ik.slerp_matrix(
                                    orientation.as_matrix(),target_rotation,ik.smoothstep(fraction)))
                            q,pe,re = ik.solve(point,quaternion,configs[side],mount,iters=110)
                            if pe>.001 or re>np.deg2rad(.3):
                                raise RuntimeError(f'{phase}: {pe:.4f} m, {re:.4f} rad')
                            configs[side] = q
                            action[7*side:7*side+6] = q
                        tick(phase)
                    if end_angles:
                        angles[:] = end_angles

                action[[6,13]] = .24
                above = [p+[0.,s*.051,.061] for p,s in zip(grasps,(1.,-1.))]
                clear = [p+[0.,s*.035,0.] for p,s in zip(grasps,(1.,-1.))]
                approach_both(above,2.8,'position both wrists outside the folded towel')
                translate_both(clear,1.8,'line up with the exposed towel rims')
                translate_both(grasps,1.6,'straddle both textile layers')
                action[[6,13]] = 0.
                hold(25,'pinch both sides of the folded towel')
                lift = .132 if cubby else .067
                translate_both([p+[0.,0.,lift] for p in grasps],3.25 if cubby else 2.0,
                               'lift the folded towel with both arms')
                hold(24,'check the suspended towel')
                suspended = trace[-20:]
                if cubby:
                    shelf_z = CUBBY_CENTER[2]+CUBBY_SIZE[2]/2
                    if min(r['cloth_z_p10_m'] for r in suspended)<shelf_z+.006:
                        raise RuntimeError('The towel needs lift clearance above the open compartment')
                    corners = array(cloth.data.nodal_pos_w)[0][:,:2]
                    middle = (corners.min(0)+corners.max(0))/2
                    shift = np.r_[CUBBY_CENTER[:2]-middle,lift]
                    target = [p+shift for p in grasps]
                    carry_angles = [Rotation.from_euler('z',-30.*s,degrees=True)*
                                    Rotation.from_euler('x',-80.*s,degrees=True)*Rotation.from_quat(DOWN_GRIP)
                                    for s in (1.,-1.)]
                    translate_both(target,3.1,'carry the folded towel over its compartment',carry_angles)
                    height = trace[-1]['cloth_z_p10_m']-shelf_z
                    descent = max(0.,min(.072,height-.021))
                    translate_both([p-[0.,0.,descent] for p in target],1.45,'bring the towel down to the shelf')
                    action[[6,13]] = .44
                    hold(22,'release the towel into the compartment')
                    translate_both([p+[0.,0.,.037] for p in target],1.55,'withdraw both grippers')
                    hold(36,'observe the towel in the compartment')
        tail = fold_tail
        width = max(row['towel_width_m'] for row in tail)
        edges_m = max(max(row['edge_x_m']) for row in tail)
        result = {'scope':'actual YAM board folding and release','controls':len(trace),'maximum_fold_deg':max(r['board_deg'] for r in trace),
                'final_width_m':width,'final_board_degrees':tail[-1]['board_deg'],'final_edge_x_max_m':edges_m,
                'folded':bool(width<=.210 and edges_m<PIVOT[0]-.065 and abs(tail[-1]['board_deg'])<=4.)}
        if pickup:
            rise = min(r['cloth_z_p10_m'] for r in suspended)-max(r['cloth_z_p10_m'] for r in fold_tail)
            result.update(suspended_rise_m=rise,suspended=rise>=.035,
                          suspended_width_m=max(r['towel_width_m'] for r in suspended))
        if cubby:
            window = trace[-20:]
            low = np.min([r['cloth_min_xyz'] for r in window],axis=0)
            high = np.max([r['cloth_max_xyz'] for r in window],axis=0)
            limits = np.array([.102,.127])
            contained = (np.all(low[:2]>=CUBBY_CENTER[:2]-limits)
                         and np.all(high[:2]<=CUBBY_CENTER[:2]+limits)
                         and low[2]>=CUBBY_CENTER[2]+.004 and high[2]<=CUBBY_CENTER[2]+.045
                         and max(r['towel_width_m'] for r in window)<=.210)
            result.update(cubby_contained=bool(contained),cubby_min_xyz_m=low.tolist(),cubby_max_xyz_m=high.tolist(),
                          complete=bool(result['folded'] and result['suspended'] and contained))
            return result | audit(trace)
        return result
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pickup',action='store_true')
    parser.add_argument('--cubby',action='store_true')
    parser.add_argument('--asset',type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dream',action='store_true')
    mode.add_argument('--setup-only',action='store_true')
    args = parser.parse_args()
    if args.dream and args.asset is None:
        parser.error('--dream requires the prebuilt --asset')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app
    try:
        from yam_arena import inspect_embodiment

        inspect_embodiment._APP = app
        cubby = args.cubby or args.asset is not None
        report = run(args.output,args.pickup or cubby,cubby,args.asset,args.dream,args.setup_only)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
