#!/usr/bin/env python3
"""The control I was missing: is rf.forward deterministic for IDENTICAL input?

Controls A/B came out at ~2.4e-5 in fp64, where a pure reordering should sit near 1e-15. That
points at the implementation itself being non-deterministic (GPU atomics / non-fixed reduction
order) rather than at arithmetic ordering. If so, every earlier number must be read against THAT
floor, not against fp precision -- and the parity check in the AWS runbook, which asserts han_db
should agree "bit-for-bit", would be unachievable by construction.
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

def fwd(q,s,tx,rx,prec):
    r=rf.forward(Scene(q=q,s=s,tx=tx,rx=rx,k=KV),precision=prec)
    try: return rf.cp.asnumpy(r)
    except Exception: return r
def rel(a,b): return float(np.abs(a-b).max()/max(np.abs(a).max(),1e-30))

print("D: IDENTICAL input, run twice  (the implementation's own noise floor)")
for prec in ("fp32","fp64"):
    r1=fwd(Q,S,TX,RX,prec); r2=fwd(Q,S,TX,RX,prec)
    same=np.array_equal(r1,r2)
    print(f"   {prec}: bit-identical={same}   max rel {rel(r1,r2):.3e}")

print("\nE: same points, order shuffled, run against a SECOND shuffle of the same order")
o=rng.permutation(len(Q))
for prec in ("fp32","fp64"):
    a1=fwd(Q[o],S[o],TX,RX,prec); a2=fwd(Q[o],S[o],TX,RX,prec)
    print(f"   {prec}: shuffled-vs-itself max rel {rel(a1,a2):.3e}")

print("\nF: float64 input arrays cast to float32 BEFORE the call (isolates host-side cast)")
Qf=Q.astype(np.float32).astype(np.float64)
for prec in ("fp32","fp64"):
    b1=fwd(Q,S,TX,RX,prec); b2=fwd(Qf,S,TX,RX,prec)
    print(f"   {prec}: fp64-coords vs fp32-rounded-coords max rel {rel(b1,b2):.3e}")
