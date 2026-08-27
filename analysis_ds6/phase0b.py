#!/usr/bin/env python3
"""PHASE 0, corrected: compare the frame change against a CONTROL that changes arithmetic order
without changing geometry.

The first run failed a 1e-5 threshold that was set without a control and is wrong for this
problem. This is interferometry at lambda ~ 1 mm over r ~ 2 m -- 2000 wavelengths -- so fp32's
~1e-7 relative precision becomes a phase error of 2*pi*2000*1e-7 ~ 1.3e-3 rad. Permuting axes
reorders dx^2+dy^2+dz^2, and fp32 addition is not associative.

CONTROLS (same geometry, different arithmetic order):
  A  shuffle the scatterer order            -- reorders the coherent sum
  B  split the point set 50/50 and re-add   -- the control used for the superposition check
TEST:
  C  permute the axes into the target frame
If C is comparable to A and B, the frame is irrelevant and the difference is pure fp32 ordering.
fp64 is run alongside: if C collapses to ~1e-15 there, nothing is axis-aware.
"""
import os, sys, time
import numpy as np
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
for p in ("/home/jonah/probe", "/home/jonah/walkthrough",
          "/home/jonah/Documents/code/model_gen/python"):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p)
import raforward as rf
from raforward import Scene, C0
from recon import ReconGrid, backproject
import body as bodylib
from stance import screening_pose
from mesh_sample import sample_mesh
import trimesh

NFREQ=128; FREQ=np.linspace(275e9,305e9,NFREQ); KV=2*np.pi*FREQ/C0
STANDOFF=2.0; NPTS=200_000
rng=np.random.default_rng(0)
g=np.linspace(-0.067/2,0.067/2,130); xx,zz=np.meshgrid(g,g,indexing="ij")
RX=np.column_stack([xx.ravel(),np.zeros(130*130),zz.ravel()]); TX=np.array([[0.15,0.0,0.0]])

m=bodylib.get_model("male")
Vp,F=bodylib.posed_verts(m,betas=np.zeros(10),body_pose=screening_pose())
me=trimesh.Trimesh(vertices=Vp,faces=F,process=False); me.fix_normals()
zc=0.5*(Vp[:,1].min()+Vp[:,1].max())
bq,_,_=sample_mesh(np.asarray(me.vertices),np.asarray(me.faces),C0/290e9/4,front_cull=False)
sel=rng.choice(len(bq),size=min(NPTS,len(bq)),replace=False); bq=np.asarray(bq)[sel]
Q=np.column_stack([bq[:,0], STANDOFF-bq[:,2], bq[:,1]-zc]).astype(np.float64)
S=rng.uniform(0.3,1.0,len(Q)).astype(np.float32)
to_t=lambda P: np.column_stack([-P[:,0],P[:,2],P[:,1]])

def fwd(q,s,tx,rx,prec):
    r=rf.forward(Scene(q=q,s=s,tx=tx,rx=rx,k=KV),precision=prec)
    try: return rf.cp.asnumpy(r)
    except Exception: return r
def rel(a,b): return float(np.abs(a-b).max()/max(np.abs(a).max(),1e-30))

for prec in ("fp32","fp64"):
    try:
        base=fwd(Q,S,TX,RX,prec)
    except Exception as e:
        print(f"\n{prec}: unavailable ({str(e)[:60]})"); continue
    # A: shuffled scatterer order, identical geometry
    o=rng.permutation(len(Q)); a=fwd(Q[o],S[o],TX,RX,prec)
    # B: 50/50 split then re-add
    h=len(Q)//2; b=fwd(Q[:h],S[:h],TX,RX,prec)+fwd(Q[h:],S[h:],TX,RX,prec)
    # C: the frame change
    c=fwd(to_t(Q),S,to_t(TX),to_t(RX),prec)
    print(f"\n{prec}  (max relative difference vs baseline)")
    print(f"   A shuffle order      : {rel(base,a):.3e}   <- control, geometry unchanged")
    print(f"   B 50/50 split re-add : {rel(base,b):.3e}   <- control, geometry unchanged")
    print(f"   C FRAME PERMUTATION  : {rel(base,c):.3e}   <- the test")
    ctrl=max(rel(base,a),rel(base,b)); tst=rel(base,c)
    print(f"   => frame effect / control = {tst/max(ctrl,1e-30):.2f}x  "
          f"{'INDISTINGUISHABLE from arithmetic noise' if tst <= 3*ctrl else 'EXCEEDS controls'}")
