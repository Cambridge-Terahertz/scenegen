"""Verify the new forward kernel on tesla: correctness vs the backed-up Jul-08 kernel,
and speed with/without phase recursion, at the dataset driver's actual shapes."""
import importlib.util, os, sys, time
import numpy as np

MG = os.path.expanduser("~/Documents/code/model_gen/python")
sys.path.insert(0, MG)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


C0 = 299_792_458.0
NFREQ, NRX, NPTS = 256, 130 * 130, 400_000
rng = np.random.default_rng(0)
g = np.linspace(-0.067 / 2, 0.067 / 2, 130)
xx, zz = np.meshgrid(g, g, indexing="ij")
RX = np.column_stack([xx.ravel(), np.zeros(NRX), zz.ravel()])
TX = np.array([[0.15, 0.0, 0.0]])
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2 * np.pi * FREQ / C0
q = np.column_stack([rng.uniform(-0.3, 0.3, NPTS), rng.uniform(1.9, 2.1, NPTS),
                     rng.uniform(-0.8, 0.8, NPTS)])
s = np.full(NPTS, 0.6)


def run(mod, recursion, reps=3):
    os.environ["RA_PHASE_RECURSION"] = "1" if recursion else "0"
    Scene = mod.Scene
    sc = Scene(q=q, s=s, tx=TX, rx=RX, k=KV)
    out = mod.forward(sc, precision="fp32")          # warm up / compile
    try:
        mod.cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    t = []
    for _ in range(reps):
        t0 = time.time()
        out = mod.forward(sc, precision="fp32")
        try:
            mod.cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        t.append(time.time() - t0)
    o = out.get() if hasattr(out, "get") else np.asarray(out)
    return min(t), o


new = load("rf_new", f"{MG}/raforward.py")
old = load("rf_old", "/tmp/raforward_jul08.py")
print(f"CuPy: new={new._HAS_CUPY} old={old._HAS_CUPY}")
print(f"scene: {NPTS:,} pts x {NRX:,} rx x {NFREQ} freq "
      f"= {NPTS*NRX*NFREQ/1e9:.1f} G evaluations\n")

t_old, o_old = run(old, False)
t_new, o_new = run(new, False)
t_rec, o_rec = run(new, True)

den = np.linalg.norm(o_old)
print(f"{'kernel':<34}{'time':>9}{'Geval/s':>11}{'speedup':>9}{'relerr vs Jul-08':>19}")
for lbl, t, o in [("Jul-08 baseline", t_old, o_old),
                  ("today: fast-math (default)", t_new, o_new),
                  ("today: + phase_recursion", t_rec, o_rec)]:
    rel = np.linalg.norm(o - o_old) / den
    print(f"{lbl:<34}{t:>8.3f}s{NPTS*NRX*NFREQ/1e9/t:>11.1f}{t_old/t:>8.2f}x{rel:>19.2e}")
