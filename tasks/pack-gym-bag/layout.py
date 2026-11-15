"""Physical task layout. Coordinates in the original living-room frame."""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
ASSETS = Path(__file__).resolve().parent / 'assets'
WORLD_ROT=np.diag([-1.,-1.,1.])
def task_pos(p): return tuple(float(v) for v in WORLD_ROT@np.asarray(p)+[.90,.705,0])
def rz(d): return Rotation.from_euler('z',d,degrees=True).as_matrix()
def ry(d): return Rotation.from_euler('y',d,degrees=True).as_matrix()
def rx(d): return Rotation.from_euler('x',d,degrees=True).as_matrix()
BOUNDS={
 'bottle':([-.036491834,-.036495917,0],[.036491834,.036495917,.2562]),
 'shoe_left':([-.14848388,-.055105824,0],[.14848388,.055105824,.104883373]),
 'shoe_right':([-.14848388,-.055105824,0],[.14848388,.055105824,.104883373]),
 'wallet':([-.057518691,-.045034424,0],[.057518691,.045034424,.01504]),
 'headphones':([-.1012,-.04173,0],[.1012,.04173,.17750])}
def corners(n):
    lo,hi=BOUNDS[n]
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
# xy denotes the rotated bounding box centre; z denotes bottom height.
ITEMS={
 'bottle':dict(xy=(.49,.08),bottom=.453,R=ry(90),mass=.25,arm='left',dest=(.070,-.110),destR=rx(90),grasp=(0,0,.0365+.013),yaw=0),
 'shoe_left':dict(xy=(.90,-.08),bottom=.019,R=rz(-90),mass=.30,arm='left',dest=(-.235,0),destR=rz(180),grasp=(0,.050,.060+.013),yaw=90),
 'shoe_right':dict(xy=(1.03,-.08),bottom=.019,R=rz(-90),mass=.30,arm='left',dest=(-.118,0),destR=rz(180),grasp=(0,.050,.060+.013),yaw=90),
 'wallet':dict(xy=(.62,1.35),bottom=.558,R=np.eye(3),mass=.12,arm='right',dest=(.250,-.100),destR=rz(180),grasp=(0,0,.01504/2+.013),yaw=0),
 'headphones':dict(xy=(.83,1.40),bottom=.558,R=rx(90),mass=.25,arm='right',dest=(.175,.075),destR=rz(-90)@rx(90),grasp=(-.089,.04375,.04173+.013),yaw=90),
}
BAG_POS=np.array([.55,.65,.182]); BAG_R=rz(90); BAG_SCALE=(1.20,1.65,1.)
def root_pose(n,xy,bottom,R,points=None):
    pts=(corners(n) if points is None else points)@R.T;mid=(pts.min(0)+pts.max(0))/2
    return np.array([xy[0]-mid[0],xy[1]-mid[1],bottom-pts[:,2].min()])
def start_pose(n):
    c=ITEMS[n];return task_pos(root_pose(n,c['xy'],c['bottom'],c['R'])),tuple(Rotation.from_matrix(WORLD_ROT@c['R']).as_quat())
def goal(n):
    c=ITEMS[n];xy=(BAG_POS+BAG_R@[*c['dest'],0])[:2]
    return root_pose(n,xy,.182+.0475+.003,c['destR']),c['destR']
