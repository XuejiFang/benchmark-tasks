"""Read true poses; send joint targets only. Save uncut video and telemetry.

No object/root pose writes or fixed grasp constraints after reset.
"""
import argparse,json,math,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from layout import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--objects',default='shoe_left,shoe_right,headphones,wallet,bottle');p.add_argument('--settle-only',action='store_true');p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=True).app
from geometry import vertices
from packing_observer import PackingObserver
import carb.settings
settings=carb.settings.get_settings();settings.set('/rtx/post/aa/op',3);settings.set('/rtx/post/dlss/execMode',2)
from runtime import install
install(args.width,args.height)
from yam_arena import inspect_embodiment,ik
inspect_embodiment._APP=app
from yam_arena.inspect_embodiment import YamArenaEmbodiment
from yam_arena.constants import READY,FINGER_OPEN,FINGER_CLOSED
from yam_arena.demo import ArmPlan,Waypoint
from yam_arena.runtime import arm_q,tcp_world
from inspect_robots import Scene
from types import SimpleNamespace
from PIL import Image,ImageDraw
E=YamArenaEmbodiment(headless=True,cam_height=args.height,cam_width=args.width,control_hz=20,task=str(Path(__file__).with_name('task.py')),object_jitter=0)
obs=E.reset(Scene(id="gym-packing-expert",instruction="Pack the gym bag."),seed=0)
base=E._base
# This file records and evaluates state directly; skip the unused callback's
# duplicate whole-scene AABB computation, leaving physics/action handling intact.
E._task_step=lambda: SimpleNamespace(observation=E._observation())
import imageio_ffmpeg
video=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{args.width}x{args.height}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'expert.mp4')],stdin=subprocess.PIPE)
top_video=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{args.width}x{args.height}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'expert_topdown.mp4')],stdin=subprocess.PIPE)
metadata={'controller':'privileged simulation-state expert','objects':args.objects.split(','),'control_hz':20,'physics_hz':240,'resolution':[args.width,args.height],'seed':0,'bag_scale':BAG_SCALE,'dream1':False,'state_writes_after_reset':False,'containment_geometry':'full visual mesh vertices','asset_sha256':{}}
for path in [ASSETS/'room/physics_room_v1.usda',ASSETS/'room/duffel_ready.usda',ASSETS/'bag/duffel_open_curved.usda',*[ASSETS/'items'/f'{n}.usd' for n in ITEMS]]:
    metadata['asset_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
log=(args.output/'trace.jsonl').open('w');step=0
observer = PackingObserver(base)
np.savez_compressed(args.output / 'scoring_vertices.npz', **observer.points)
action=np.array([*READY,1,*READY,1],float)
def arr(x):return x.torch.detach().cpu().numpy() if hasattr(x,'torch') else x.detach().cpu().numpy()
def state(n):
    d=base.scene.rigid_objects[n].data
    pose=arr(d.root_pose_w)[0]
    return pose[:3],Rotation.from_quat(pose[3:]).as_matrix(),arr(d.root_lin_vel_w)[0],arr(d.root_ang_vel_w)[0]
def capture(obs,label):
    global step
    row = observer.capture(action, step)
    row['phase'] = label
    score = row['score']
    log.write(json.dumps(row)+'\n');log.flush()
    img=Image.fromarray(obs.images['top_cam'].copy());draw=ImageDraw.Draw(img)
    draw.rectangle((0,0,args.width,43),fill=(10,10,10));draw.text((9,5),'PRIVILEGED STATE SCRIPTED EXPERT | PHYSICS SIMULATION',fill='white');draw.text((9,23),f'{step/20:5.1f}s | {label} | verified packing score: {score:.3f}',fill='white')
    video.stdin.write(np.asarray(img).tobytes())
    overhead=arr(base.scene.sensors['overview_cam'].data.output['rgb'])[0,:,:,:3].astype(np.uint8)
    top_video.stdin.write(overhead.tobytes())
    if step%20==0:Image.fromarray(overhead).save(args.output/'latest_topdown.png')
    if step%20==0:img.save(args.output/'latest.png');print('STEP',step,label,'SCORE',score,flush=True)
    step+=1
    return row

def tick(label):
    global obs
    if step >= 3000:
        raise RuntimeError('Exceeded the 3,000-step control budget')
    result=E.step(SimpleNamespace(data=action));obs=result.observation
    return capture(obs,label)
try:
    for _ in range(40):tick('initial settling')
    Image.fromarray(obs.images['top_cam']).save(args.output/'initial.png')
    if not args.settle_only:
        for n in args.objects.split(','):
            c=ITEMS[n];side=c['arm'];idx=0 if side=='left' else 7
            pos,R,_,_=state(n)
            # Rotate the planned grasp offset with the observed object yaw.
            sourceR=WORLD_ROT@c['R'];deltaR=R@sourceR.T
            pts=corners(n)@R.T+pos; centre=(pts.min(0)+pts.max(0))/2
            bottom=pts[:,2].min();offset=WORLD_ROT@np.array(c['grasp']);offset=deltaR@offset
            pick=centre.copy();pick[:2]+=offset[:2];pick[2]=bottom+offset[2]
            destpos,destR=goal(n);destpts=corners(n)@destR.T+destpos;destcentre=(destpts.min(0)+destpts.max(0))/2
            rotate=destR@c['R'].T;destoffset=rotate@np.array(c['grasp']);place=destcentre.copy();place[:2]+=destoffset[:2];place[2]=destpts[:,2].min()+destoffset[2]+.004;place=np.array(task_pos(place))
            yaw=math.radians(c['yaw']+180)+math.atan2(deltaR[1,0],deltaR[0,0]);end_yaw=yaw+math.atan2(rotate[1,0],rotate[0,0])
            travel=max(pick[2]+.075,.66 if c['arm']=='left' else .64)
            from feedback_control import pick_plan,transport_plan
            basepos=(0.,.355 if side=='left' else -.355,.55)
            actions=pick_plan(pick,yaw,travel,basepos,arm_q(base.scene[side+'_robot']))
            print('PICK_PLAN',n,'pick',pick,flush=True)
            for a,label in actions:
                action[idx:idx+7]=a;tick(n+': '+label)
            op,orr,_,_=state(n)
            if op[2]<pos[2]+.04:raise RuntimeError(n+': object did not lift; abort')
            desired=[]
            if n=='bottle':
                worldR=WORLD_ROT.T@orr
                aligned=Rotation.align_vectors([c['destR'][:,2]],[worldR[:,2]])[0].as_matrix()
                c=dict(c,destR=aligned@worldR)
            rotations=[rz(flip)@c['destR'] for flip in (0,180)]
            if n=='headphones':
                # Natural earcup/headband resting attitudes, not a forced flat
                # bounding box. Search several feasible physical placements.
                rotations=[rz(yaw)@rx(pitch) for pitch in (102.5,100,105,90,110) for yaw in (180,0,90,-90)]
            for dr in rotations:
                xy=(BAG_POS+BAG_R@[*c['dest'],0])[:2]
                dp=root_pose(n,xy,.182+.0475+.007,dr,vertices(n))
                desired.append((task_pos(dp),WORLD_ROT@dr))
            actions,planinfo=transport_plan(arm_q(base.scene[side+'_robot']),basepos,op,orr,desired,.54 if n=="bottle" else (.65 if n.startswith("shoe") else .59),lower_at_pick=n=="bottle",prefer_front=n.startswith("shoe"),object_points=corners(n))
            print('PLACE_PLAN',n,planinfo,flush=True)
            for a,label in actions:
                action[idx:idx+7]=a;tick(n+': '+label)
            # Retract to actual rig ready pose with joint interpolation.
            start=action[idx:idx+6].copy()
            for i in range(60):
                s=ik.smoothstep((i+1)/60);action[idx:idx+6]=start+s*(np.array(READY)-start);tick(n+': return to ready')
    for _ in range(60):last=tick('final released settling')
    (args.output/'result.json').write_text(json.dumps({'score':last['score'],'success':last['score']==1.,'physics_control_only':True,'steps':step,'objects':last['objects']},indent=2))
except Exception as exc:
    (args.output/'failure.json').write_text(json.dumps({'step':step,'error':str(exc)},indent=2))
    raise
finally:
    log.close();video.stdin.close();video.wait();top_video.stdin.close();top_video.wait();E.close();app.close()
