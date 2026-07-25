import bpy, numpy as np, glob, os, sys
d=sys.argv[-2]; out=sys.argv[-1]; os.makedirs(out,exist_ok=True)
files=sorted(glob.glob(d+"/PocketKnife_*.fbx"))
def mesh_pts(dg,meshes):
    pts=[]
    for o in meshes:
        ev=o.evaluated_get(dg); m=ev.to_mesh(); mw=o.matrix_world
        pts+=[list(mw @ v.co) for v in m.vertices]; ev.to_mesh_clear()
    return np.array(pts)
print(f"\n=== exporting closed pose for {len(files)} knives ===")
for f in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    nm=os.path.basename(f).replace('.fbx','')
    sc=bpy.context.scene
    if bpy.data.actions:
        fr=bpy.data.actions[0].frame_range; f0,f1=int(fr[0]),int(fr[1])
    else: f0=f1=sc.frame_current
    # scan frames, pick most compact (min longest extent = closed)
    best=(1e9,f0)
    for fr in list(range(f0,f1+1,max(1,(f1-f0)//14)))+[f1]:
        sc.frame_set(fr); dg=bpy.context.evaluated_depsgraph_get()
        P=mesh_pts(dg,meshes)
        if len(P)==0: continue
        L=float((P.max(0)-P.min(0)).max())
        if L<best[0]: best=(L,fr)
    sc.frame_set(best[1]); dg=bpy.context.evaluated_depsgraph_get()
    P=mesh_pts(dg,meshes); D=np.sort(P.max(0)-P.min(0))[::-1]
    # export OBJ at closed frame (armature applied)
    for o in bpy.data.objects: o.select_set(o.type=='MESH')
    bpy.context.view_layer.objects.active=meshes[0]
    op=os.path.join(out,nm+"_closed.obj")
    bpy.ops.wm.obj_export(filepath=op,export_selected_objects=True,apply_modifiers=True)
    print(f"{nm}: closed@frame {best[1]}  bbox {D[0]:.3f}x{D[1]:.3f}x{D[2]:.3f}  long/mid={D[0]/D[1]:.2f}")
