"""Process-local articulated-prop extension for the existing dual-YAM adapter.

Spawn the source composition unchanged as a scene asset, not a RigidObject.
PhysX retains its bodies, joints, springs and limits. Observation uses live
body views; only the robot adapter receives actions after reset.
Import after Isaac's AppLauncher has started.
"""
from pathlib import Path
import numpy as np


def array(value):
    value = value.torch if hasattr(value, 'torch') else value
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.array(value, copy=True)


def install(usd, output, position, rotation_xyzw, contact_bodies,
            room_factory=None, robot_contact_offset=.0005, track_friction=False):
    from pxr import Usd, UsdGeom, UsdPhysics
    from yam_arena import scene, embodiment
    from yam_arena.robot import asset_link_paths
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab.sensors import ContactSensorCfg

    # A wrapper overrides only imported scene clocks. Geometry/physics remain
    # references to the verified source package; no scaling or body freezing.
    wrapper = Path(output) / 'mechanism.usda'
    stage = Usd.Stage.CreateNew(str(wrapper))
    root = UsdGeom.Xform.Define(stage, '/Mechanism').GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(Path(usd).resolve()))
    disabled = []
    for prim in list(Usd.PrimRange(root)):
        if prim.IsA(UsdPhysics.Scene):
            disabled.append(str(prim.GetPath()))
            prim.SetActive(False)
    stage.GetRootLayer().Save()
    original_room = scene.task_room

    def room(*args):
        surrounding = original_room(*args) if room_factory is None else room_factory(*args)
        return [*surrounding, Object(
            name='mechanism', object_type=ObjectType.BASE, usd_path=str(wrapper),
            initial_pose=Pose(position_xyz=tuple(position), rotation_xyzw=tuple(rotation_xyzw)),
            spawn_cfg_addon={'activate_contact_sensors': True})]

    scene.task_room = room
    original_embodiment = embodiment.BimanualYamEmbodiment

    def make_embodiment(*args, **kwargs):
        # 2 mm buttons need a contact envelope below their stroke, not 5 mm.
        kwargs['robot_contact_offset'] = robot_contact_offset
        return original_embodiment(*args, **kwargs)

    embodiment.BimanualYamEmbodiment = make_embodiment
    original_build = scene.build_task_environment
    links = asset_link_paths()

    def build(*args, **kwargs):
        arena = original_build(*args, **kwargs)
        previous = arena.env_cfg_callback

        def configure(cfg):
            cfg = previous(cfg)
            for name, relative_path in contact_bodies.items():
                setattr(cfg.scene, 'mechanism_contact_' + name, ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/mechanism' + relative_path,
                    filter_prim_paths_expr=[
                        '{ENV_REGEX_NS}/' + side + links[link]
                        for side in ('LeftArm', 'RightArm')
                        for link in ('gripper', 'tip_left', 'tip_right')],
                    history_length=0, track_contact_points=True,
                    track_friction_forces=track_friction,
                    max_contact_data_count_per_prim=256))
            return cfg

        arena.env_cfg_callback = configure
        return arena

    scene.build_task_environment = build
    return {'wrapper': str(wrapper), 'disabled_physics_scenes': disabled,
            'robot_contact_offset_m': robot_contact_offset, 'source_scale': 1.0}


class MechanismObserver:
    """Read articulated state through PhysX without introducing state setters."""

    def __init__(self, stage, root='/World/envs/env_0/mechanism'):
        from pxr import Usd, UsdPhysics
        import omni.physics.tensors as tensors
        self.root = root
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise RuntimeError('Missing mechanism root: ' + root)
        self.physics = tensors.create_simulation_view('torch')
        self.physics.set_subspace_roots('/')
        self.bodies = {}
        self.joints = {}
        for child in Usd.PrimRange(prim):
            path = str(child.GetPath())
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                view = self.physics.create_rigid_body_view(path)
                if view.count != 1:
                    raise RuntimeError('Expected one live body at ' + path)
                self.bodies[path[len(root):]] = view
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.joints[path[len(root):]] = self.physics.create_articulation_view(path)
        if not self.bodies:
            raise RuntimeError('Mechanism has no live rigid bodies')

    def read(self):
        self.physics.update_articulations_kinematic()
        bodies = {name: {'pose_xyzw': array(v.get_transforms())[0].tolist(),
                         'velocity': array(v.get_velocities())[0].tolist()}
                  for name, v in self.bodies.items()}
        joints = {name: {'positions': array(v.get_dof_positions())[0].tolist(),
                         'velocities': array(v.get_dof_velocities())[0].tolist(),
                         'names': list(v.shared_metatype.dof_names)}
                  for name, v in self.joints.items() if v.count}
        for body in bodies.values():
            if not np.isfinite(body['pose_xyzw'] + body['velocity']).all():
                raise RuntimeError('Non-finite mechanism state')
        return {'bodies': bodies, 'articulations': joints}
