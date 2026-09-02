# dataset7 forward-model worker for AWS Batch (GPU).
# Built ON tesla with podman, where model_gen and the threat_dataset modules already live, so we
# COPY them from the build context instead of git-cloning private repos (no deploy key needed).
#
# GPU arch: CuPy for CUDA 12.x runs on A10G (sm_86, g5), L4 (sm_89, g6), L40S (sm_89, g6e).
# NOT tesla's local sm_120 build. The forward model and backprojection are fp32 throughout, so a
# mid-tier GPU is plenty -- the win comes from running 32 of them, not from a bigger one.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip awscli && rm -rf /var/lib/apt/lists/*

# Keep these as SEPARATE pip installs. A single --index-url would force cupy to resolve from the
# torch CPU index, where it does not exist -- the failure that cost a rebuild on the July image.
RUN pip3 install --no-cache-dir numpy scipy trimesh smplx cupy-cuda12x
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The driver's sys.path probes ~/walkthrough and ~/Documents/code/model_gen/python. In-container
# we run as root, so ~ = /root and dropping the code at those exact paths needs no symlinks.
COPY threat_dataset/   /root/walkthrough/
COPY model_gen_python/ /root/Documents/code/model_gen/python/
COPY ds7/              /root/ds7/
COPY entrypoint.py     /opt/entrypoint.py

# Assets (SMPL-X models, objects_v5 meshes, glock/phone STLs, shapes_10x10.json) are
# license-restricted and pulled from private S3 at runtime into /root -- see entrypoint.py.
ENTRYPOINT ["python3", "/opt/entrypoint.py"]
