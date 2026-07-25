import bpy, numpy as np, sys
# clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)
fbx=sys.argv[-1]
bpy.ops.import_scene.fbx(filepath=fbx)
meshes=[o for o in bpy.data.objects if o.type=='MESH']
print(f"\n=== FBX imported: {len(meshes)} mesh object(s) ===")
allpts=[]
for o in sorted(meshes,key=lambda o:-len(o.data.vertices)):
    mw=o.matrix_world
    pts=np.array([list(mw @ v.co) for v in o.data.vertices])
    if len(pts)==0: continue
    allpts.append(pts); d=pts.max(0)-pts.min(0)
    print(f"  {o.name[:34]:34s} verts={len(pts):7d}  dims=({d[0]:.2f},{d[1]:.2f},{d[2]:.2f})")
if allpts:
    P=np.vstack(allpts); D=np.sort(P.max(0)-P.min(0))[::-1]
    print(f"\nOVERALL L/M/T {D[0]:.2f} x {D[1]:.2f} x {D[2]:.2f}  long/mid={D[0]/D[1]:.2f} (open~high, closed~2-3)")
    print(f"separable blade? -> {'YES, '+str(len(allpts))+' objects' if len(allpts)>1 else 'NO, single fused mesh'}")
