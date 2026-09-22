#!/usr/bin/env python3
"""
Record what an image's PyTorch build is, and diff two records.

  python build_manifest.py record --out hermetic.json   # inside each image
  python build_manifest.py diff hermetic.json upstream.json

Two images at the same PyTorch commit can still differ in compiler, CPU
target, math library, NCCL, glibc, Python, or how the container was
launched. `diff` prints the fields that differ; that list is where a
performance difference between them can come from. No GPU needed, ~1 min.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

# Every Nth disassembled line is sampled when measuring the instruction
# mix; a prime avoids aliasing with any periodicity in the output, and 37
# keeps a 335 MB .text section under two seconds.
ISA_SAMPLE_STRIDE = 37

# Instruction classes worth knowing about: a build targeting a newer
# -march emits these in the dispatch and CPU-op paths.
ISA_PATTERNS = {
    "avx512f": r"\bzmm\d",
    "avx512_vnni": r"\bvpdpbusd",
    "avx2": r"\bvpermd|\bvpbroadcastd",
    "fma": r"\bvfmadd",
    "amx": r"\btdpb",
}


def sh(*cmd, text=True):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       text=text, timeout=120).strip()
    except Exception:
        return None


def declared():
    import torch
    cfg = torch.__config__.show()
    out = {
        "torch_version": torch.__version__,
        "torch_commit": getattr(torch.version, "git_version", None),
        "cuda": torch.version.cuda,
        "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version()))
                if torch.cuda.is_available() else None,
        "python": sys.version.replace("\n", " "),
        "glibc": platform.libc_ver()[1],
        "driver": (sh("nvidia-smi", "--query-gpu=driver_version",
                      "--format=csv,noheader") or "").split("\n")[0],
        "config_show": cfg,
    }
    # config_show is a bulleted "built with" list followed by a block of
    # "KEY : value" settings; different builds populate different subsets,
    # so pull both shapes and keep whatever is there.
    for label, pat in [("compiler", r"^\s*-\s*((?:GCC|Clang|clang|MSVC)[^\n]*)"),
                       ("cxx_standard", r"C\+\+ Version:\s*(\d+)"),
                       ("blas", r"^\s*-\s*(Intel\(R\) oneAPI Math Kernel[^\n]*|"
                                r"OpenBLAS[^\n]*|LAPACK[^\n]*|BLAS_INFO[^\n]*)"),
                       ("openmp", r"^\s*-\s*(OpenMP[^\n]*)"),
                       ("onednn", r"MKL-DNN (v[\d.]+)")]:
        m = re.search(pat, cfg, re.M)
        if m:
            out[label] = m.group(1).strip()
    for key in ("CXX_FLAGS", "TORCH_VERSION", "CMAKE_BUILD_TYPE", "USE_CUDNN",
                "USE_NCCL", "USE_MKLDNN", "USE_OPENMP", "BUILD_TYPE"):
        m = re.search(rf"^\s*{key}\s*:\s*(.*)$", cfg, re.M)
        if m and m.group(1).strip():
            out[key.lower()] = m.group(1).strip()
    m = re.search(r"_GLIBCXX_USE_CXX11_ABI=(\d)", cfg)
    if m:
        out["cxx11_abi"] = m.group(1)

    freeze = sh(sys.executable, "-m", "pip", "freeze")
    if freeze:
        out["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()[:16]
        out["pip_freeze_lines"] = len(freeze.splitlines())
    return out


def binary():
    import torch
    libdir = os.path.join(os.path.dirname(torch._C.__file__), "lib")
    out = {"libdir": libdir, "libs": {}}
    for name in ("libtorch_cpu.so", "libtorch_cuda.so"):
        path = os.path.join(libdir, name)
        if not os.path.exists(path):
            continue
        info = {"bytes": os.path.getsize(path)}

        comment = sh("readelf", "-p", ".comment", path)
        if comment:
            info["producers"] = sorted({
                line.split("]", 1)[1].strip()
                for line in comment.splitlines() if "]" in line and line.strip()
            })

        text = sh("size", "-A", path)
        if text:
            m = re.search(r"^\.text\s+(\d+)", text, re.M)
            if m:
                info["text_bytes"] = int(m.group(1))

        # GCC records the command line only with -frecord-gcc-switches, so
        # treat its absence as unknown rather than scraping .rodata, where
        # unrelated text yields every -O level at once.
        switches = sh("readelf", "-p", ".GCC.command.line", path)
        if switches:
            flags = sorted({
                tok for line in switches.splitlines()
                for tok in re.findall(r"-(?:O[0-3s]|march=[\w.-]+|mtune=[\w.-]+|"
                                      r"flto[\w=-]*|DNDEBUG|fopenmp)", line)
            })
            info["gcc_switches"] = flags

        ldd = sh("ldd", path)
        if ldd:
            info["links"] = sorted({
                m.group(1) for m in re.finditer(r"(lib(?:nccl|mkl|openblas|blis|"
                                                r"omp|gomp|cudart|cublas)[\w.+-]*\.so[\w.]*)",
                                                ldd)
            })

        if shutil.which("objdump"):
            # Which ISA the build targets, as a fraction of sampled
            # instructions: PyTorch ships runtime-dispatched AVX-512 kernels
            # in every build, so presence alone says nothing -- the share
            # does. Sample evenly across .text rather than its head, where
            # one compilation unit would dominate.
            counts = sh("bash", "-c",
                        f"objdump -d --no-show-raw-insn --section=.text {path} 2>/dev/null "
                        f"| awk 'NR%{ISA_SAMPLE_STRIDE}==0'")
            if counts:
                total = counts.count("\n") or 1
                info["isa_share"] = {
                    k: round(len(re.findall(pat, counts)) / total, 4)
                    for k, pat in ISA_PATTERNS.items()
                }
                info["isa_sampled_lines"] = total
        out["libs"][name] = info
    return out


def runtime():
    cpuinfo = ""
    try:
        cpuinfo = open("/proc/cpuinfo").read()
    except OSError:
        pass
    m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    gov = sh("bash", "-c",
             "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")
    out = {
        "cpu_model": m.group(1).strip() if m else None,
        "cpu_count": os.cpu_count(),
        "allowed_cpus": len(os.sched_getaffinity(0)),
        "governor": gov,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "kernel": platform.release(),
        "os_release": sh("bash", "-c",
                         "grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2"),
        "numa_nodes": sh("bash", "-c",
                         "ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l"),
        "gpu_clocks": sh("nvidia-smi",
                         "--query-gpu=name,clocks.sm,clocks.mem,persistence_mode",
                         "--format=csv,noheader"),
        "container_hints": {
            k: v for k, v in os.environ.items()
            if k in ("HOSTNAME", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                     "NCCL_DEBUG", "LD_PRELOAD", "LD_LIBRARY_PATH")
        },
    }
    return out


def cmd_record(args):
    try:
        import torch  # noqa: F401
    except Exception as e:
        sys.exit(f"cannot import torch in this environment: {e}")
    missing = [t for t in ("readelf", "objdump", "size", "ldd")
               if not shutil.which(t)]
    if missing:
        print(f"note: {', '.join(missing)} not found; those binary fields "
              f"will be absent (install binutils for the full manifest)",
              file=sys.stderr)
    manifest = {"declared": declared(), "binary": binary(), "runtime": runtime()}
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    d = manifest["declared"]
    print(f"torch {d['torch_version']} ({(d['torch_commit'] or '')[:12]})  "
          f"cuda {d['cuda']}  nccl {d['nccl']}")
    for name, info in manifest["binary"]["libs"].items():
        isa = info.get("isa_share", {})
        top = " ".join(f"{k}={v:.3f}" for k, v in isa.items() if v > 0) or "-"
        print(f"  {name}: {info['bytes'] / 1e6:.0f} MB  text "
              f"{info.get('text_bytes', 0) / 1e6:.0f} MB  {top}")
        for prod in info.get("producers", []):
            print(f"      {prod}")
    print(f"written {args.out}")


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        yield prefix, ", ".join(map(str, obj))
    else:
        yield prefix, obj


def load_manifest(path):
    try:
        m = json.load(open(path))
    except (OSError, ValueError) as e:
        sys.exit(f"{path}: cannot read manifest ({e})")
    if not isinstance(m, dict) or "declared" not in m:
        sys.exit(f"{path}: not a manifest written by `record`")
    # The raw config blob is long; compare it as a digest and let the
    # extracted fields say what actually differs.
    m.setdefault("declared", {})["config_show"] = hashlib.sha256(
        m["declared"].get("config_show", "").encode()).hexdigest()[:16]
    return m


def cmd_diff(args):
    a, b = load_manifest(args.a), load_manifest(args.b)
    fa, fb = dict(flatten(a)), dict(flatten(b))
    keys = sorted(set(fa) | set(fb))
    if not keys:
        sys.exit("both manifests are empty")
    width = min(max(len(k) for k in keys), 48)
    name_a, name_b = os.path.basename(args.a)[:34], os.path.basename(args.b)[:34]
    rows = [(k, fa.get(k, "(absent)"), fb.get(k, "(absent)"))
            for k in keys if fa.get(k) != fb.get(k)]

    print(f"{'field':<{width}}  {name_a:<34}  {name_b}")
    print("-" * (width + 72))
    for k, va, vb in rows:
        print(f"{k[:width]:<{width}}  {str(va)[:34]:<34}  {str(vb)[:34]}")
    print(f"\n{len(rows)} field(s) differ, {len(keys) - len(rows)} identical")

    same_commit = (a["declared"].get("torch_commit")
                   == b["declared"].get("torch_commit")
                   and a["declared"].get("torch_commit"))
    if rows and same_commit:
        print("\nSame torch commit, so a performance difference between these "
              "images comes from one of the fields above. Change one at a "
              "time to attribute it.")
    elif rows:
        print("\nDifferent torch commits: the PyTorch source differs too, not "
              "only the fields above.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="write this image's manifest")
    r.add_argument("--out", default="build_manifest.json")
    d = sub.add_parser("diff", help="diff two manifests")
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args()
    (cmd_record if args.cmd == "record" else cmd_diff)(args)


if __name__ == "__main__":
    main()
