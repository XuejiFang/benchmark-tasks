"""Actual visual-mesh vertices for containment, without empty OBB corners."""
from functools import lru_cache
import numpy as np
from pxr import Usd,UsdGeom
from layout import ASSETS
@lru_cache(None)
def vertices(name):
    stage=Usd.Stage.Open(str(ASSETS/'items'/f'{name}.usd'))
    root=stage.GetDefaultPrim();cache=UsdGeom.XformCache();root_inv=cache.GetLocalToWorldTransform(root).GetInverse();cloud=[]
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            pts=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=np.float64)
            if len(pts):
                m=np.asarray(cache.GetLocalToWorldTransform(p)*root_inv)
                cloud.append(pts@m[:3,:3]+m[3,:3])
    if not cloud:raise RuntimeError('No visual mesh for '+name)
    return np.concatenate(cloud)
