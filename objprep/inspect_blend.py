import bpy, numpy as np
meshes=[o for o in bpy.data.objects if o.type=='MESH']
print(f"\n=== {len(meshes)} mesh object(s) ===")
allpts=[]
rows=[]
for o in meshes:
    mw=o.matrix_world
    pts=np.array([list(mw @ v.co) for v in o.data.vertices])
    if len(pts)==0: continue
    allpts.append(pts)
    d=pts.max(0)-pts.min(0)
    rows.append((o.name,len(pts),d))
for n,nv,d in sorted(rows,key=lambda r:-r[1]):
    print(f"  {n[:34]:34s} verts={nv:7d}  dims=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f})")
if allpts:
    P=np.vstack(allpts); D=np.sort(P.max(0)-P.min(0))[::-1]
    print(f"\nOVERALL bbox sorted L/M/T: {D[0]:.3f} x {D[1]:.3f} x {D[2]:.3f}")
    print(f"  long/mid = {D[0]/D[1]:.2f}   long/thin = {D[0]/D[2]:.2f}")
    print(f"  -> OPEN folder ~ long/mid 4-6 (blade extended); CLOSED ~ long/mid 2-3")
