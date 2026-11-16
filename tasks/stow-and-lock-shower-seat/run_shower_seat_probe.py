"""Measure the pinned fold-down shower seat under gravity and applied hinge torque."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


def add_catch(stage, root='/World/SeatCatch', stiffness=.0035):
    from pxr import Gf, UsdPhysics
    from build_key_return_scene import box, rigid
    from build_tea_probe import contact

    origin = (.246, -.106, .825)
    stage.DefinePrim(root, 'Xform')
    box(stage, root+'/WallBracket', (.258, -.050, .824), (.024, .086, .026), (.30, .34, .37))
    bar = root+'/Bar'
    rigid(stage, bar, .034, origin)
    box(stage, bar+'/Lever', (0., 0., .036), (.015, .010, .075), (.78, .42, .19))
    contact(stage.GetPrimAtPath(bar+'/Lever'), .00045)
    pin = UsdPhysics.RevoluteJoint.Define(stage, root+'/Keeper')
    pin.CreateAxisAttr('Y')
    pin.CreateBody1Rel().SetTargets([bar])
    pin.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    pin.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
    pin.CreateLowerLimitAttr(-90.)
    pin.CreateUpperLimitAttr(0.)
    release = UsdPhysics.DriveAPI.Apply(pin.GetPrim(), 'angular')
    release.CreateStiffnessAttr(stiffness)
    release.CreateDampingAttr(.0012)
    release.CreateTargetPositionAttr(-90.)
    return bar


def run(source, output, use_catch, stiffness):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    import omni.physics.tensors as tensors
    from pxr import Sdf, Usd, UsdPhysics
    import torch
    from articulated_fixture import array

    dt = 1/480
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device='cpu', use_fabric=False, physics=PhysxCfg(enable_ccd=True)))
    root = '/World/ShowerSeat'
    stage = sim.stage
    prop = stage.DefinePrim(root, 'Xform')
    prop.GetReferences().AddReference(str(source.resolve()))
    for layer in stage.GetUsedLayers():
        if Path(layer.identifier).name=='animation.usda':
            stage.MuteLayer(layer.identifier)
    for prim in list(Usd.PrimRange(prop)):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
    hinge = stage.GetPrimAtPath(root+'/Physics/Joints/fold')
    seat_path = root+'/Geometry/Installation/Seat'
    seat_prim = stage.GetPrimAtPath(seat_path)
    if not hinge.IsA(UsdPhysics.RevoluteJoint) or not seat_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('Expected the source fold joint and seat body')
    seat_prim.AddAppliedSchema('PhysxRigidBodyAPI')
    seat_prim.CreateAttribute('physxRigidBody:disableGravity', Sdf.ValueTypeNames.Bool).Set(False)
    seat_mass = UsdPhysics.MassAPI(seat_prim).GetMassAttr().Get()
    catch_path = add_catch(stage,stiffness=stiffness) if use_catch else None
    sim.reset()
    physics = tensors.create_simulation_view('torch', stage_id=sim_utils.get_current_stage_id())
    physics.set_subspace_roots('/')
    joints = [physics.create_articulation_view(str(p.GetPath()))
              for p in Usd.PrimRange(prop) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    live = [joint for joint in joints if joint.count==1 and 'fold' in joint.shared_metatype.dof_names]
    seat = physics.create_rigid_body_view(seat_path)
    if len(live)!=1 or seat.count!=1:
        raise RuntimeError('Expected one live source seat articulation')
    joint = live[0]
    catch = physics.create_rigid_body_view(catch_path) if use_catch else None
    if use_catch and catch.count!=1:
        raise RuntimeError('Expected one physical seat catch')
    axis = list(joint.shared_metatype.dof_names).index('fold')
    index = torch.zeros(1, dtype=torch.int32, device=sim.device)
    no_force = torch.zeros((1,3), device=sim.device)
    trace = []
    for tick in range(4320):
        time = tick*dt
        angle = float(np.rad2deg(array(joint.get_dof_positions())[0,axis]))
        torque = 0.
        if catch is not None and time < 5.8:
            catch.apply_forces_and_torques_at_position(
                no_force, torch.tensor([[0.,.44,0.]], device=sim.device), None, index, True)
        if 1. <= time < 6.5:
            rate = array(seat.get_velocities())[0,3]
            torque = float(np.clip(-3.3*(1.2+rate)-12.6*np.cos(np.deg2rad(angle)), -16.0, 3.0))
        elif 7.2 <= time < 7.6:
            torque = .60
        if torque:
            seat.apply_forces_and_torques_at_position(
                no_force, torch.tensor([[torque,0.,0.]], device=sim.device), None, index, True)
        sim.step(render=False)
        if tick % 24 == 23:
            trace.append({'time_s': (tick+1)*dt, 'fold_degrees': angle, 'torque_nm': torque,
                          'seat_pose_xyzw': array(seat.get_transforms())[0].tolist(),
                          'catch_pose_xyzw': array(catch.get_transforms())[0].tolist() if catch is not None else None})
    (output/'trace.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in trace))
    folded = [row['fold_degrees'] for row in trace if row['time_s'] >= 8.0]
    return {'scope': 'source shower seat hinge loaded by external torque and then released',
            'physical_catch': use_catch,
            'seat_mass_kg': seat_mass, 'initial_settled_angle_deg': trace[17]['fold_degrees'],
            'maximum_angle_deg': max(row['fold_degrees'] for row in trace),
            'max_fixture_torque_nm': max(abs(row['torque_nm']) for row in trace),
            'released_min_angle_deg': min(folded), 'stayed_upright': min(folded)>=85.}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--catch', action='store_true')
    parser.add_argument('--catch-stiffness',type=float,default=.0035)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cpu').app
    try:
        result = run(args.source,args.output,args.catch,args.catch_stiffness)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
