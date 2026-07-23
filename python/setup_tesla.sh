#!/usr/bin/env bash
# Bootstrap the model_gen Python/CuPy port on a fresh GPU box.
#
# Target: tesla (RTX 5090, 32 GB, Blackwell sm_120) — but nothing here is
# tesla-specific except the defaults; it works on bose (A4000, sm_86) too.
#
#   scp setup_tesla.sh tesla:~/ && ssh tesla 'bash ~/setup_tesla.sh'
#
# Idempotent: safe to re-run. Each step is skipped if already satisfied.
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:Cambridge-Terahertz/model_gen.git}"
REPO_DIR="${REPO_DIR:-$HOME/Documents/code/model_gen}"
BRANCH="${BRANCH:-python-port}"
ENV_NAME="${ENV_NAME:-ra}"
PY_VER="${PY_VER:-3.11}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# tesla has NO NAS mount (unlike bose's /mnt/shared_drive) -> results go local.
OUTPUT_ROOT="${RA_OUTPUT_ROOT:-$HOME/synthetic_data}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. GPU check
say "1/6  GPU + driver"
command -v nvidia-smi >/dev/null || { echo "FATAL: no nvidia-smi — not a GPU box"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# CUDA major decides the CuPy wheel. Prefer nvcc (the real toolkit), fall back to
# the driver's max-supported version reported by nvidia-smi.
if command -v nvcc >/dev/null; then
  CUDA_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  echo "nvcc toolkit: CUDA ${CUDA_MAJOR}.x"
else
  CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
  echo "no nvcc; driver supports up to CUDA ${CUDA_MAJOR}.x"
fi
case "$CUDA_MAJOR" in
  13) CUPY_PKG="cupy-cuda13x[ctk]" ;;
  12) CUPY_PKG="cupy-cuda12x[ctk]" ;;
  *)  echo "FATAL: unhandled CUDA major '${CUDA_MAJOR}' — pick a wheel by hand"; exit 1 ;;
esac
echo "-> CuPy wheel: ${CUPY_PKG}"
# The [ctk] extra is NOT optional: without the bundled toolkit headers CuPy's
# NVRTC compile of the forward kernel dies with "Failed to find CUDA headers".

# ------------------------------------------------------------- 2. miniconda
say "2/6  miniconda"
if [ ! -d "$CONDA_DIR" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_DIR"
  rm -f /tmp/mc.sh
else
  echo "already at $CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
# Fresh miniconda refuses to solve until the channel ToS is accepted.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# ------------------------------------------------------------------- 3. env
say "3/6  conda env '${ENV_NAME}' (python ${PY_VER})"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "already exists"
else
  conda create -y -n "$ENV_NAME" "python=${PY_VER}"
fi
conda activate "$ENV_NAME"

say "3b/6  python deps"
# Everything via pip so the CuPy wheel and its deps stay consistent.
pip install -q --upgrade pip
pip install -q numpy scipy matplotlib h5py pillow pytest trimesh networkx
pip install -q "$CUPY_PKG"
python -c "import cupy; cupy.arange(4).sum(); print('cupy', cupy.__version__, 'OK on', cupy.cuda.Device(0).compute_capability)"

# ------------------------------------------------------------------ 4. repo
say "4/6  repo @ ${BRANCH}"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --all --quiet
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only
else
  mkdir -p "$(dirname "$REPO_DIR")"
  # needs SSH agent forwarding (`ssh -A tesla` / ForwardAgent yes) for the GitHub key
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1

# ------------------------------------------------------- 5. output root (!)
say "5/6  RA_OUTPUT_ROOT"
# results.py defaults to /mnt/shared_drive/... (the bose NAS mount). That path
# does not exist here, so every run would fail at save time. Bake the override
# into the env so it applies to every future `conda activate ra`.
mkdir -p "$OUTPUT_ROOT"
ACT_D="$CONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACT_D"
printf 'export RA_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT" > "$ACT_D/ra_output_root.sh"
export RA_OUTPUT_ROOT="$OUTPUT_ROOT"
echo "runs will be saved under $RA_OUTPUT_ROOT"

# ------------------------------------------------------------ 6. smoke test
say "6/6  smoke test"
cd "$REPO_DIR/python"
python -m pytest -q raforward.py recon.py
# small synthetic run: exercises forward -> recon -> save end to end
python run_model.py --n-points 200000 --run-id smoke_$(hostname)
echo
echo "SETUP OK. Next session:  conda activate ${ENV_NAME} && cd ${REPO_DIR}/python"
