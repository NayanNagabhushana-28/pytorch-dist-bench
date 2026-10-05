# nccl-tests as a control

`nccl-tests` (github.com/NVIDIA/nccl-tests) measures collectives with no
PyTorch in the path. It is the control for any claim about a PyTorch build:
if two images differ in `bench_collectives` but agree in `nccl-tests`, the
difference is in PyTorch or its launch environment; if both differ, it is
NCCL, CUDA, the driver, or the node.

We do not vendor it: it must be built against *each image's own* NCCL and
CUDA to be a control at all, it moves independently of this suite, and a
copy in this repo would be stale and would test the wrong library.

Build and run inside each image:

```bash
git clone https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests
make -C /tmp/nccl-tests -j MPI=0 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Same sizes as bench_collectives: 1 KB .. 1 GB, x4 per step.
for coll in all_reduce all_gather reduce_scatter; do
  /tmp/nccl-tests/build/${coll}_perf -b 1K -e 1G -f 4 -g 8 -n 1000 -w 200 \
    | tee "nccl-tests_${coll}.txt"
done
```

Record alongside the numbers: `python build_manifest.py record` from the
same container, and the NCCL version the binary linked
(`ldd /tmp/nccl-tests/build/all_reduce_perf | grep nccl`).

Reading the output against ours:

| nccl-tests column | our metric | note |
|---|---|---|
| `time (us)` at 1 KB | `alpha` | ours is larger: it includes c10d dispatch and a per-call sync. `eager - graph` from `bench_allreduce_dispatch` is roughly that difference. |
| `algbw` at 1 GB | `beta` (`algo_bw_gbps`) | directly comparable |
| `busbw` | `bus_bw_gbps` | same formula, `2(n-1)/n` for all-reduce |

Interpretation:

- nccl-tests and ours both differ between images -> below PyTorch.
- nccl-tests agree, ours differ -> PyTorch build or launch environment;
  `build_manifest.py diff` lists the candidates.
- nccl-tests differ, ours agree -> a path we do not exercise (algorithm or
  protocol selection at sizes our sweep skips).

Run it interleaved with the PyTorch runs, in one session, like any other
A/B. A difference measured in separate sessions is not attributable.
