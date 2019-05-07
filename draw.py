import cv2
import numpy as np
import random

def draw_circle(h,w,xy_list,r=5):
    rz=np.zeros((len(xy_list),1,h,w),dtype=np.uint8)
    switch=[lambda x,h:x,lambda x,h:h-x-1]
    for i,(x,y) in enumerate(xy_list):
        rx=random.randint(0,1)
        ry=random.randint(0,1)
        tx=switch[rx](x,w)
        ty=switch[ry](y,h)
        cv2.circle(rz[i][0],(tx,ty), r, 255, -1)
    rz=rz.astype(np.float32)*2/255-1
    return rz
def rand_draw(h,w,nub):
    xx=np.random.randint(0,high=h,size=nub)
    yy=np.random.randint(0,high=w,size=nub)
    xy_l=np.append(xx,yy).reshape((2,-1)).T
    rand_norm=np.random.normal(size=(nub,8,h,w))
    xy_l2=xy_l.copy().astype(np.float32)
    xy_l2[:,0]=xy_l2[:,0]*2/w-1
    xy_l2[:,1]=xy_l2[:,1]*2/h-1
    real_in=np.concatenate((rand_norm,xy_l2.reshape((-1,2,1,1)).repeat(h, axis=2).repeat(w, axis=3)),axis=1).astype(np.float32)
    xy_label=xy_l2.reshape((-1,2,1,1)).repeat(h, axis=2).repeat(w, axis=3)
    rz=draw_circle(h,w,xy_l)
    return rz,xy_label,real_in
def c_draw(h,w):
    xx=np.asarray([128])
    yy=np.asarray([128])
    xy_l=np.append(xx,yy).reshape((2,-1)).T
    rand_norm=np.random.normal(size=(1,8,h,w))
    xy_l2=xy_l.copy().astype(np.float32)
    xy_l2[:,0]=xy_l2[:,0]*2/w-1
    xy_l2[:,1]=xy_l2[:,1]*2/h-1
    real_in=np.concatenate((rand_norm,xy_l2.reshape((-1,2,1,1)).repeat(h, axis=2).repeat(w, axis=3)),axis=1).astype(np.float32)
    xy_label=xy_l2.reshape((-1,2,1,1)).repeat(h, axis=2).repeat(w, axis=3)
    rz=draw_circle(h,w,xy_l)
    return rz,xy_label,real_in
