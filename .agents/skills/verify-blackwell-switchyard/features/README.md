# Verification map

Run the parent skill's launch and doctor first. Use a fresh CLI process per drive,
the existing environment, and an external evidence directory. Capture command,
exit status, observable output, and result files. Keep proof after cleanup.
CPU-only commands explicitly hide GPUs. GPU recipes require a free device and
the project's guard; DDP requires both devices.

- [Operator correctness](correctness.md): reference API, production forward and
  backward, batched inference, and isolated experimental plans.
- [Performance evidence](performance.md): framework/native baselines, implementation
  identity, timing, measured limits, and backward campaign acceptance.
- [Training integration](training.md): arena/stack equivalence, fixed-batch training,
  full-step measurement, and two-GPU gradient validation.

Run only checks affected by the change or needed to fill a demonstrated gap.
Report coverage per entry point. Passing a CPU test does not verify a GPU entry
point; compilation does not establish runtime correctness or speed. Record the
attempted route and unmet prerequisite for an unreachable feature.
