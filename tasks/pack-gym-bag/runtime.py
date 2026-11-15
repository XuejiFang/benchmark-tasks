"""Process-local Arena adapter; never changes the installed YAM package."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from layout import *

def install(width=960,height=540,full_resolution_wrists=False):
    from yam_arena import station, scene, embodiment
    from yam_arena.rig import Rig,Camera,Lens
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
    import isaaclab.sim as sim_utils
    station.BASES={'left':(0.,.355,.55),'right':(0.,-.355,.55)}
    eye=np.array(task_pos((2.70,-2.25,1.60))); target=np.array(task_pos((.65,.60,.38)))
    z=(eye-target);z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    cam=Camera(tuple(eye),tuple(Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat()),'opengl',Lens.from_focal(25.6,width,height))
    station.rig=lambda top_cam='calibrated':Rig('station-livingroom',station.BASES,cam,{s:station.wrist_camera(s) for s in station.BASES},True)
    original_emb=embodiment.BimanualYamEmbodiment
    original_camera_cfg=original_emb._camera_cfg
    def cameras(self,height,width,overview):
        cfg=original_camera_cfg(self,height,width,True)
        cfg.left_cam=self._wrist_cam_cfg('left',height if full_resolution_wrists else 180,width if full_resolution_wrists else 320)
        cfg.right_cam=self._wrist_cam_cfg('right',height if full_resolution_wrists else 180,width if full_resolution_wrists else 320)
        cfg.overview_cam.offset=embodiment.CameraCfg.OffsetCfg(pos=task_pos((.74,.65,2.60)),rot=(0.,0.,-2**-.5,2**-.5),convention='opengl')
        cfg.overview_cam.spawn.focal_length=24.
        return cfg
    original_emb._camera_cfg=cameras
    def make_emb(*a,**kw):
        kw.update(cam_width=width,cam_height=height)
        return original_emb(*a,**kw)
    embodiment.BimanualYamEmbodiment=make_emb
    def room(*_):
        return [Object(name='room',object_type=ObjectType.BASE,usd_path=str(ASSETS/'room/physics_room_v1.usda'),initial_pose=Pose(position_xyz=(.90,.705,0.),rotation_xyzw=(0.,0.,1.,0.)))]
    scene.task_room=room
    def item(spec):
        if spec.name=='gym_bag':
            path=ASSETS/'room/duffel_ready.usda'; pos=task_pos(BAG_POS); rot=tuple(Rotation.from_matrix(WORLD_ROT@BAG_R).as_quat());scale=BAG_SCALE
            addon=dict(physics_material=scene._grippy(),func=scene._spawn_fixture,rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True))
        else:
            path=ASSETS/'items'/f'{spec.name}.usd';pos,rot=start_pose(spec.name);scale=(1.,1.,1.)
            addon=dict(physics_material=scene._grippy(),mass_props={'':[sim_utils.MassCfg(mass=ITEMS[spec.name]['mass'])]},mass_props_create_if_missing=True,collision_props=scene._contact_props())
        return Object(name=spec.name,usd_path=str(path),object_type=ObjectType.RIGID,scale=scale,initial_pose=Pose(position_xyz=tuple(map(float,pos)),rotation_xyzw=tuple(map(float,rot))),spawn_cfg_addon=addon)
    scene.task_object=item
    original_build=scene.build_task_environment
    def build(*args,**kwargs):
        arena=original_build(*args,**kwargs);callback=arena.env_cfg_callback
        sensors={}
        from yam_arena.robot import asset_link_paths
        links=asset_link_paths()
        for n in ITEMS:
            cfg=arena.scene.assets[n].get_contact_sensor_cfg(contact_against_object=arena.scene.assets['gym_bag'])
            cfg.filter_prim_paths_expr += ['{ENV_REGEX_NS}/'+side+links[link] for side in ('LeftArm','RightArm') for link in ('gripper','tip_left','tip_right')]
            cfg.track_friction_forces=True;cfg.track_contact_points=True;cfg.max_contact_data_count_per_prim=256
            sensors['bag_contact_'+n]=cfg
        def configure(cfg):
            cfg=callback(cfg)
            for n,sensor in sensors.items():setattr(cfg.scene,n,sensor)
            return cfg
        arena.env_cfg_callback=configure
        return arena
    scene.build_task_environment=build
