"""Privileged grasp-relative placement through ordinary joint motor targets."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik
from yam_arena.constants import READY,FINGER_OPEN,FINGER_CLOSED
from yam_arena.demo import ArmPlan,Waypoint,grip_rotation

def interpolate(qstart,base,goals):
    q=np.array(qstart,float);prevpos,prevrot=ik.tcp_pose(q,base);out=[]
    for duration,pos,rot,grip,label in goals:
        pos=np.array(pos);maxpe=maxre=0.
        for i in range(round(duration*20)):
            s=ik.smoothstep((i+1)/round(duration*20));target=prevpos+s*(pos-prevpos);r=ik.slerp_matrix(prevrot,rot,s)
            q,pe,re=ik.solve(target,ik.matrix_to_quat_xyzw(r),q,base_pos=base,iters=60)
            maxpe=max(maxpe,pe);maxre=max(maxre,re)
            if pe>.005 or re>math.radians(3):raise RuntimeError(f'{label}: IK {pe*1000:.1f}mm {math.degrees(re):.1f}deg')
            out.append((np.r_[q,grip],label))
        prevpos,prevrot=pos,rot
    return out

def pick_plan(pick,yaw,travel,base,qstart):
    for flip in (0.,math.pi):
        y=yaw+flip;hover=np.array([*pick[:2],travel]);R=grip_rotation(y)
        q0,pe,re=ik.solve(hover,ik.matrix_to_quat_xyzw(R),np.array(READY),base_pos=base,iters=300)
        if pe>.003 or re>.03:continue
        try:
            out=[(np.r_[qstart+ik.smoothstep((i+1)/60)*(q0-qstart),1.],'approach') for i in range(60)]
            out+=interpolate(q0,base,[(2.5,pick,R,1.,'descend'),(1.,pick,R,0.,'close'),(2.5,hover,R,0.,'lift'),(.5,hover,R,0.,'hold grasp')])
            return out
        except RuntimeError:continue
    raise RuntimeError('No feasible grasp approach')

def transport_plan(qstart,base,objpos,objR,desired_poses,travel,lower_at_pick=False,prefer_front=False,object_points=None):
    tcp,Rtcp=ik.tcp_pose(qstart,base)
    relR=Rtcp.T@objR;relp=Rtcp.T@(objpos-tcp)
    errors=[]
    for dp,dr in desired_poses:
        Rdest=dr@relR.T;pdest=np.array(dp)-Rdest@relp
        hover=np.array([*pdest[:2],max(travel,pdest[2]+.20)])
        front_routes=tuple('front_'+str(v) for v in (22,24,26,20,18,28,30,34,38,40))
        routes=(*front_routes,'direct') if prefer_front else ('direct',*front_routes)
        for route in routes:
            goals=[]
            start_height=tcp[2]
            if lower_at_pick and tcp[2]>travel+.03:
                start_height=travel
                goals.append((2.,np.array([*tcp[:2],travel]),Rtcp,0.,'adjust transfer height'))
            if route.startswith('front_'):
                # Move away from the shoulder before crossing between sofa and bag.
                clear_x=float(route.split('_')[1])/100
                midpoint=np.array([max(clear_x,tcp[0]),tcp[1],start_height])
                goals.append((2.,midpoint,Rtcp,0.,'clear shoulder'))
                midpoint2=np.array([max(clear_x,hover[0]),hover[1],hover[2]])
                goals.append((3.,midpoint2,Rdest,0.,'carry around shoulder'))
            goals.extend([(3.,hover,Rdest,0.,'carry'),(3.,pdest,Rdest,0.,'lower'),(1.,pdest,Rdest,1.,'release'),(2.,hover,Rdest,1.,'retreat')])
            try:
                actions=interpolate(qstart,base,goals)
                if object_points is not None:
                    for a,label in actions:
                        if label in ('lower','release','retreat'):continue
                        tp,tr=ik.tcp_pose(a[:6],base)
                        pts=np.asarray(object_points)@(tr@relR).T+(tp+tr@relp)
                        lo,hi=pts.min(0),pts.max(0)
                        for mount_y in (-.355,.355):
                            overlap=lo[0]<.10 and hi[0]>-.10 and lo[1]<mount_y+.10 and hi[1]>mount_y-.10
                            if overlap and lo[2]<.56 and hi[2]>.52:
                                raise RuntimeError('transport intersects robot mounting plate clearance')
                return actions,dict(target_tcp=pdest.tolist(),route=route)

            except RuntimeError as e:errors.append(str(e))
    raise RuntimeError('Placement paths rejected: '+ '; '.join(errors))
