import bpy, numpy as np, glob, os, sys
d=sys.argv[-1]
files=sorted(glob.glob(d+"/PocketKnife_*.fbx"))
print(f"\n=== {len(files)} pocket knife FBX ===")
print(f"{'knife':10s}{'meshes':>7}{'arm':>4}{'verts':>8}{'L/M/T (bbox)':>22}{'long/mid':>9}{'anim frames':>13}")
for f in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    arms=[o for o in bpy.data.objects if o.type=='ARMATURE']
    pts=[]
    for o in meshes:
        mw=o.matrix_world
        pts += [list(mw @ v.co) for v in o.data.vertices]
    nm=os.path.basename(f).replace('PocketKnife_','').replace('.fbx','')
    if not pts: print(f"{nm:10s}  (no verts)"); continue
    P=np.array(pts); D=np.sort(P.max(0)-P.min(0))[::-1]
    acts=bpy.data.actions
    fr=""
    if acts:
        a=acts[0]; fr=f"{int(a.frame_range[0])}-{int(a.frame_range[1])}"
    print(f"{nm:10s}{len(meshes):>7}{len(arms):>4}{len(P):>8}"
          f"{D[0]:>7.2f}x{D[1]:.2f}x{D[2]:.2f}{D[0]/max(D[1],1e-6):>9.2f}{fr:>13}")
