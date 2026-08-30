# blackwell-switchyard

High-performance **Block Attention Residuals** (Block AttnRes) for NVIDIA Blackwell GPUs.

> **Status: early.** The repository is being built in reviewable stages. Nothing is claimed
> here that has not been measured on the hardware named below. This banner is removed once
> the headline result is in place.

Attention Residuals ([Kimi Team, arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) replaces the
fixed additive residual connection with softmax attention over the outputs of preceding layers, so
each layer aggregates earlier representations with learned, input-dependent weights. **Block
AttnRes** is the practical variant that partitions depth into blocks to bound the cost.

Computationally the operator is a *depth-wise* attention: for every token position independently,
a softmax runs over the block axis rather than the sequence axis. That gives it a very different
performance profile from ordinary attention, and it is the profile this repository is about.

## What this project measures

| Path | Purpose |
|------|---------|
| Reference | Paper-faithful PyTorch. Readable, slow, and the correctness oracle. |
| Framework baseline | The strongest formulation PyTorch + `torch.compile`/Inductor can reach. |
| Switchyard | The optimized execution path, selected by profiling rather than by preference. |

Headline results, the shapes they were measured at, and the reproduction commands go here once
they exist and have survived independent review.

## Hardware under test

NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, compute capability 12.0 (`sm_120`),
188 SMs, 128 MiB L2, 95 GiB usable, 300 W. Two such GPUs, connected over PCIe (no NVLink).
Full capture in [`docs/machine.md`](docs/machine.md).

## Project state

See [`PROJECT_STATE.md`](PROJECT_STATE.md) for what is done, in progress, and next, and for the
decision log.

## Attribution

Attention Residuals is the work of the **Kimi Team at Moonshot AI**
([paper](https://arxiv.org/abs/2603.15031), [code](https://github.com/MoonshotAI/Attention-Residuals)).
This repository is an independent performance-engineering implementation and is not affiliated with
Moonshot AI. See [`NOTICE`](NOTICE) for attribution details and
[`docs/`](docs/) for the citation.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
