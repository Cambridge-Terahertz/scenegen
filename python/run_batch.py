"""Batch-process every complete pose folder under a scene root.

    python run_batch.py "/mnt/shared_drive/Jonah/synthetic_data/26JUN19 2"
    python run_batch.py <root> --assets gun phone --limit 5
    python run_batch.py <root> --redo            # reprocess even if already saved

Scans <root> for pose folders that have body/body.xyz (so partially-uploaded
poses are skipped), runs forward->recon->save for each into
<RA_OUTPUT_ROOT>/<out-name>/<pose>, skips poses already done (re-runnable as
more finish uploading), and writes a contact-sheet index at the end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import raforward as rf
from bodies import load_real_scene
from results import run_dir, output_root, contact_sheet
from run_model import process_and_save


def find_complete_poses(root) -> list:
    """Pose subfolders that have body/body.xyz, natural-sorted (pose_2 < pose_10)."""
    root = Path(root)
    poses = [p for p in root.iterdir()
             if p.is_dir() and (p / "body" / "body.xyz").exists()]
    return sorted(poses, key=lambda p: (len(p.name), p.name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_root", help="folder containing pose_* subfolders")
    p.add_argument("--out-name", default="results_26JUN19",
                   help="subfolder under RA_OUTPUT_ROOT for the batch results")
    p.add_argument("--assets", nargs="*", default=["phone"])
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--no-zbuffer", action="store_true")
    p.add_argument("--zbuffer-pixel", type=float, default=0.003)
    p.add_argument("--no-density-weight", action="store_true")
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--nrx", type=int, default=16_900)
    p.add_argument("--nf", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="process at most N poses")
    p.add_argument("--redo", action="store_true", help="reprocess even if output exists")
    args = p.parse_args()

    poses = find_complete_poses(args.scene_root)
    if args.limit:
        poses = poses[:args.limit]
    print(f"found {len(poses)} complete pose(s) under {args.scene_root}\n")

    t_all = time.perf_counter()
    done = skipped = failed = 0
    for i, pose in enumerate(poses, 1):
        run_id = f"{args.out_name}/{pose.name}"
        if run_dir(run_id).exists() and not args.redo:
            print(f"[{i}/{len(poses)}] {pose.name}: already done, skip")
            skipped += 1
            continue
        print(f"[{i}/{len(poses)}] {pose.name} -> {run_id}")
        try:
            q, s = load_real_scene(pose, decimate=args.decimate, assets=args.assets,
                                   zbuffer=not args.no_zbuffer,
                                   zbuffer_pixel=args.zbuffer_pixel,
                                   density_weight=not args.no_density_weight)
            process_and_save(q, s, run_id=run_id, source=str(pose),
                             nrx=args.nrx, nf=args.nf, precision=args.precision,
                             voxel=args.voxel, extra_meta={"pose": pose.name})
            done += 1
        except Exception as e:                          # keep going on a bad pose
            print(f"  FAILED: {e}")
            failed += 1
        print()

    print(f"batch: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.perf_counter() - t_all) / 60:.1f} min")

    # campaign index across this batch's results
    try:
        sheet = contact_sheet(output_root() / args.out_name,
                              output_root() / args.out_name / "index.png",
                              preview="preview/model_vs_recon.png")
        print(f"contact sheet -> {sheet}")
    except Exception as e:
        print(f"contact sheet skipped: {e}")


if __name__ == "__main__":
    main()
