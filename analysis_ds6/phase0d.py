#!/usr/bin/env python3
"""Which frame is more accurate in fp32?  Measure each against its own fp64 truth.

Distances are formed as dx^2 + dy^2 + dz^2. The CURRENT frame puts range (the ~2 m term) in the
MIDDLE slot; the TARGET frame puts it LAST. Summing small+small+LARGE preserves more of the small
terms than small+LARGE+small, so the target frame may be marginally more accurate in fp32.
Same geometry, same point order -- only the axis labelling differs -- so fp64 is the truth for both.
"""
import os, sys
import numpy as np
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
for p in ("/home/jonah/probe","/home/jonah/walkthrough","/home/jonah/Documents/code/model_gen/python"):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0,p)
import raforward as rf
from raforward import Scene, C0
import body as bodylib
from stance import screening_pose
from mesh_sample import sample_mesh
import trimesh

NFREQ=128; FREQ=np.linspace(275e9,305e9,NFREQ); KV=2*np.pi*FREQ/C0
STANDOFF=2.0; rng=np.random.default_rng(0)
g=np.linspace(-0.067/2,0.067/2,130); xx,zz=np.meshgrid(g,g,indexing="ij")
RX=np.column_stack([xx.ravel(),np.zeros(130*130),zz.ravel()]); TX=np.array([[0.15,0.0,0.0]])
m=bodylib.get_model("male"); Vp,F=bodylib.posed_verts(m,betas=np.zeros(10),body_pose=screening_pose())
me=trimesh.Trimesh(vertices=Vp,faces=F,process=False); me.fix_normals()
zc=0.5*(Vp[:,1].min()+Vp[:,1].max())
bq,_,_=sample_mesh(np.asarray(me.vertices),np.asarray(me.faces),C0/290e9/4,front_cull=False)
sel=rng.choice(len(bq),size=200_000,replace=False); bq=np.asarray(bq)[sel]
Q=np.column_stack([bq[:,0],STANDOFF-bq[:,2],bq[:,1]-zc]).astype(np.float64)
S=rng.uniform(0.3,1.0,len(Q)).astype(np.float32)
to_t=lambda P: np.column_stack([-P[:,0],P[:,2],P[:,1]])

def fwd(q,tx,rx,prec):
    r=rf.forward(Scene(q=q,s=S,tx=tx,rx=rx,k=KV),precision=prec)
    try: return rf.cp.asnumpy(r)
    except Exception: return r
def rel(a,b): return float(np.abs(a-b).max()/max(np.abs(a).max(),1e-30))

cur32,cur64 = fwd(Q,TX,RX,"fp32"), fwd(Q,TX,RX,"fp64")
Qt,TXt,RXt = to_t(Q),to_t(TX),to_t(RX)
tgt32,tgt64 = fwd(Qt,TXt,RXt,"fp32"), fwd(Qt,TXt,RXt,"fp64")

e_cur, e_tgt = rel(cur64,cur32), rel(tgt64,tgt32)
print("fp32 error vs its own fp64 truth (same geometry, same point order):")
print(f"   CURRENT frame (range in the MIDDLE slot): {e_cur:.3e}")
print(f"   TARGET  frame (range in the LAST slot)  : {e_tgt:.3e}")
print(f"   ratio current/target = {e_cur/max(e_tgt,1e-30):.2f}x")
print(f"   -> {'TARGET is more accurate' if e_tgt < e_cur else 'CURRENT is more accurate' if e_cur < e_tgt else 'equivalent'} in fp32")
print(f"\n   fp64 agreement between the two frames: {rel(cur64,tgt64):.3e}  (the invariance proof)")
print(f"   in dB, the fp32 frame difference is {20*np.log10(1+rel(cur32,tgt32)):.4f} dB")
