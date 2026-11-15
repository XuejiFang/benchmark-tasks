"""Measure the source monitor slide under gravity and with a passive counterbalance."""

import argparse
import json
from pathlib import Path
import traceback


def add_height_pin(stage,root,stops, *, rail_path=None):
    """Add a spring-loaded pin that physically catches between three rack shelves."""
    from pxr import Gf, UsdGeom, UsdPhysics

    from build_key_return_scene import box, rigid
    from build_tea_probe import contact

    hinge = root+'/Links/hinge'
    pin_path = root+'/Links/height_pin'
    # The stand is fixed to the worktop. Use a separate static rail so pin
    # contact does not depend on enabling self-collisions in the source rig.
    rail = rail_path or root+'_LockRail'
    x,y,z = -.086,.087,.406
    centre = Gf.Vec3d(x,y,z)
    rigid(stage,pin_path,.021,centre)
    barrel = UsdGeom.Cylinder.Define(stage,pin_path+'/SteelPin')
    barrel.CreateAxisAttr('Y')
    barrel.CreateRadiusAttr(.004)
    barrel.CreateHeightAttr(.032)
    barrel.CreateDisplayColorAttr([Gf.Vec3f(.58,.62,.66)])
    contact(barrel.GetPrim(),.0003)
    box(stage,pin_path+'/PullHead',(0.,-.016,0.),(.017,.007,.015),(.81,.36,.15))
    rail_bottom,rail_top = z+min(stops)-.021,z+max(stops)+.017
    slots = [(z+s-.006,z+s+.006) for s in sorted(stops)]
    bounds = [rail_bottom,*(v for gap in slots for v in gap),rail_top]
    cache = UsdGeom.XformCache()
    for i,(a,b) in enumerate(zip(bounds[::2],bounds[1::2])):
        path = f'{rail}/Shelf{i}'
        box(stage,path,(x,.104,(a+b)/2),(.020,.022,b-a),(.25,.27,.30))
        contact(stage.GetPrimAtPath(path),.0003)
    spine = (x-.015,.104,(rail_bottom+rail_top)/2)
    box(stage,rail+'/Spine',spine,(.010,.022,rail_top-rail_bottom),(.22,.24,.26))
    hinge_from_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(hinge)).GetInverse()
    guide = hinge_from_world.Transform(Gf.Vec3d(x,.061,z))
    box(stage,hinge+'/PinGuide',guide,(.017,.037,.013),(.30,.32,.35))
    joint = UsdPhysics.PrismaticJoint.Define(stage,root+'/Joints/height_pin')
    joint.CreateAxisAttr('Y')
    joint.CreateLowerLimitAttr(-.028)
    joint.CreateUpperLimitAttr(0.)
    joint.CreateBody0Rel().SetTargets([hinge])
    joint.CreateBody1Rel().SetTargets([pin_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(hinge_from_world.Transform(centre)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
    joint.CreateCollisionEnabledAttr(False)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
    spring.CreateStiffnessAttr(55.)
    spring.CreateDampingAttr(3.)
    spring.CreateTargetPositionAttr(0.)
    return pin_path


def run(args):
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
    import omni.physics.tensors as tensors
    from pxr import Usd,UsdPhysics
    import torch

    from articulated_fixture import array

    dt = 1/480
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt,device='cpu',use_fabric=False,physics=PhysxCfg(enable_ccd=True)))
    stage = sim.stage
    root = '/World/MonitorStand'
    component = stage.DefinePrim(root,'Xform')
    component.GetReferences().AddReference(str(args.source.resolve()))
    for prim in list(Usd.PrimRange(component)):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
    height = stage.GetPrimAtPath(root+'/Joints/height_slide')
    limits = UsdPhysics.PrismaticJoint(height)
    if not limits:
        raise RuntimeError('Source stand has no height slide')
    moving = [stage.GetPrimAtPath(root+'/Links/'+link) for link in ('monitor','hinge')]
    moving_mass = sum(UsdPhysics.MassAPI(link).GetMassAttr().Get() for link in moving)
    for link in moving:
        link.GetAttribute('physxRigidBody:disableGravity').Set(False)
    low,high = limits.GetLowerLimitAttr().Get(),limits.GetUpperLimitAttr().Get()
    balance_point = (low+high)/2
    if args.pin:
        pin_path = add_height_pin(stage,root,(low,balance_point,high))
        moving_mass += .021
    if args.balance or args.pin:
        spring = UsdPhysics.DriveAPI.Get(height,'linear')
        spring.GetStiffnessAttr().Set(55.)
        spring.GetDampingAttr().Set(40.)
        spring.GetTargetPositionAttr().Set(balance_point+moving_mass*9.81/55.)

    anchor = UsdPhysics.FixedJoint.Define(stage,'/World/HoldStandRoot')
    anchor.CreateBody1Rel().SetTargets([root+'/Links/base'])
    sim.reset()
    physics = tensors.create_simulation_view('torch',stage_id=sim_utils.get_current_stage_id())
    physics.set_subspace_roots('/')
    monitor = physics.create_articulation_view(root)
    if monitor.count!=1:
        raise RuntimeError('Expected a single live monitor articulation')
    names = list(monitor.shared_metatype.dof_names)
    height_axis = names.index('height_slide')
    if args.pin:
        pin = physics.create_rigid_body_view(pin_path)
        screen = physics.create_rigid_body_view(root+'/Links/monitor')
        if pin.count!=1 or screen.count!=1 or 'height_pin' not in names:
            raise RuntimeError('The locking pin is missing from the live articulation')
        indices = torch.zeros(1,dtype=torch.int32,device=sim.device)

        def apply_force(part,force):
            push = torch.tensor([force],device=sim.device)
            part.apply_forces_and_torques_at_position(push,torch.zeros_like(push),None,indices,True)

    rows = []
    for step in range(4800 if args.pin else 2880):
        time = (step+1)*dt
        if args.pin:
            if args.unlock and 1.<=time<6.:
                apply_force(pin,[0.,-3.2,0.])
            if 7.<=time<8.5:
                apply_force(screen,[0.,0.,-5.])
        sim.step(render=False)
        if (step+1)%24==0:
            values = array(monitor.get_dof_positions())[0]
            rate = array(monitor.get_dof_velocities())[0]
            rows.append({'time_s':time,'positions':dict(zip(names,values.tolist())),
                         'velocities':dict(zip(names,rate.tolist()))})
            if args.pin:
                rows[-1]['pin_world_xyz'] = array(pin.get_transforms())[0,:3].tolist()
    (args.output/'trace.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    end = rows[-1]['positions'][names[height_axis]]
    rate = rows[-1]['velocities'][names[height_axis]]
    result = {'scope':'pinned source stand under gravity, without robot or lock','counterbalanced':args.balance,
            'moving_mass_kg':moving_mass,'joint_limits_m':[low,high],
            'start_height_m':rows[0]['positions']['height_slide'],'end_height_m':end,
            'end_speed_m_s':rate,'balanced_near_midstroke':abs(end-balance_point)<.008 and abs(rate)<.002}
    if args.pin:
        load_window = [r for r in rows if 7.<=r['time_s']<=8.5]
        pin_travel = min(r['positions']['height_pin'] for r in rows)
        loaded_error = max(abs(r['positions']['height_slide']-(balance_point if args.unlock else high))
                           for r in load_window)
        final_pin = rows[-1]['positions']['height_pin']
        result.update(scope='physical pin, rail and screen load without robot',counterbalanced=True,
                      unlock_command=args.unlock,pin_travel_mm=1000*pin_travel,
                      loaded_height_error_mm=1000*loaded_error,final_pin_mm=1000*final_pin,
                      locked_under_load=loaded_error<=.004 and final_pin>=-.004)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--balance',action='store_true')
    parser.add_argument('--pin',action='store_true')
    parser.add_argument('--unlock',action='store_true')
    args = parser.parse_args()
    if args.unlock and not args.pin:
        parser.error('--unlock requires --pin')
    args.output.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,device='cpu').app
    try:
        result = run(args)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        (args.output/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
