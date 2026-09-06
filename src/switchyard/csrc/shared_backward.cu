#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;
constexpr int kRegisterThreads = 512;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kThreads / kWarpSize;
constexpr int kRegisterWarps = kRegisterThreads / kWarpSize;
constexpr int kMaxLocalSources = 16;
constexpr int kStatFields = 5;
constexpr int kFeatureClusterFields = 5;

enum StatField : int {
  kQueryDot = 0,
  kGradDot = 1,
  kRstd = 2,
  kAlpha = 3,
  kDlogit = 4,
};

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value);

template <>
__device__ __forceinline__ float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value);

template <>
__device__ __forceinline__ __half from_float(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ void warp_sum3(float& x, float& y, float& z) {
  constexpr unsigned kFullMask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    x += __shfl_down_sync(kFullMask, x, offset);
    y += __shfl_down_sync(kFullMask, y, offset);
    z += __shfl_down_sync(kFullMask, z, offset);
  }
}

__device__ __forceinline__ float warp_sum(float value) {
  constexpr unsigned kFullMask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

template <typename scalar_t>
struct PairOps;

template <>
struct PairOps<__half> {
  __device__ __forceinline__ static float2 unpack(uint32_t bits) {
    union {
      uint32_t bits;
      __half2 pair;
    } value{bits};
    return __half22float2(value.pair);
  }

  __device__ __forceinline__ static uint32_t pack(float x, float y) {
    union {
      uint32_t bits;
      __half2 pair;
    } value{};
    value.pair = __floats2half2_rn(x, y);
    return value.bits;
  }
};

template <>
struct PairOps<__nv_bfloat16> {
  __device__ __forceinline__ static float2 unpack(uint32_t bits) {
    union {
      uint32_t bits;
      __nv_bfloat162 pair;
    } value{bits};
    return __bfloat1622float2(value.pair);
  }

  __device__ __forceinline__ static uint32_t pack(float x, float y) {
    union {
      uint32_t bits;
      __nv_bfloat162 pair;
    } value{};
    value.pair = __floats2bfloat162_rn(x, y);
    return value.bits;
  }
};

template <int ClusterBlocks>
__device__ __forceinline__ int cluster_rank() {
  if constexpr (ClusterBlocks == 1) {
    return 0;
  } else {
    return static_cast<int>(cg::this_cluster().block_rank());
  }
}

template <int ClusterBlocks>
__device__ __forceinline__ void cluster_sync() {
  if constexpr (ClusterBlocks == 1) {
    __syncthreads();
  } else {
    cg::this_cluster().sync();
  }
}

template <int ClusterBlocks>
__device__ __forceinline__ float* map_stats(float* local, int owner) {
  if constexpr (ClusterBlocks == 1) {
    return local;
  } else {
    return cg::this_cluster().map_shared_rank(local, owner);
  }
}

template <typename scalar_t, int ClusterBlocks>
__device__ void shared_backward_body(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_values,
    float* __restrict__ grad_query,
    int n_sources,
    int n_tokens,
    int width,
    float eps,
    int64_t stride_source,
    int64_t stride_token,
    int64_t stride_feature) {
  extern __shared__ __align__(16) unsigned char storage[];

  const int rank = cluster_rank<ClusterBlocks>();
  const int token = static_cast<int>(blockIdx.x) / ClusterBlocks;
  if (token >= n_tokens) {
    return;
  }

  const int local_capacity = (n_sources + ClusterBlocks - 1) / ClusterBlocks;
  const int source_begin = rank * local_capacity;
  const int local_sources = max(0, min(local_capacity, n_sources - source_begin));
  scalar_t* shared_values = reinterpret_cast<scalar_t*>(storage);
  const size_t value_bytes = static_cast<size_t>(local_capacity) * width * sizeof(scalar_t);
  scalar_t* shared_grad = shared_values + static_cast<int64_t>(local_capacity) * width;
  const size_t grad_bytes = static_cast<size_t>(width) * sizeof(scalar_t);
  const size_t stats_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  float* stats = reinterpret_cast<float*>(storage + stats_offset);
  float* reduction = stats + kStatFields * local_capacity;

  for (int feature = threadIdx.x; feature < width; feature += blockDim.x) {
    shared_grad[feature] = grad_out[static_cast<int64_t>(token) * width + feature];
  }
  __syncthreads();

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);

  // Spread each wave of at most eight sources across all eight warps. A full
  // wave gives one warp to each source. A partial wave gives several warps to
  // a source, so N=9 does not leave seven warps idle for its final source.
  // Every warp writes one private partial; all source reductions then finish
  // after one block barrier instead of two barriers per source.
  for (int source_base = 0; source_base < local_sources; source_base += kMaxWarps) {
    const int wave_sources = min(kMaxWarps, local_sources - source_base);
    const int source_offset = warp % wave_sources;
    const int source_warp = warp / wave_sources;
    const int source_warps = 1 + (kMaxWarps - 1 - source_offset) / wave_sources;
    const int local_source = source_base + source_offset;
    const int source = source_begin + local_source;
    float ssq = 0.0f;
    float query_dot = 0.0f;
    float grad_dot = 0.0f;
    for (int feature = source_warp * kWarpSize + lane;
         feature < width;
         feature += source_warps * kWarpSize) {
      const int64_t value_offset =
          static_cast<int64_t>(source) * stride_source +
          static_cast<int64_t>(token) * stride_token +
          static_cast<int64_t>(feature) * stride_feature;
      const scalar_t raw = values[value_offset];
      shared_values[static_cast<int64_t>(local_source) * width + feature] = raw;
      const float value = to_float(raw);
      ssq += value * value;
      query_dot += value * to_float(query[feature]);
      grad_dot += value * to_float(shared_grad[feature]);
    }
    warp_sum3(ssq, query_dot, grad_dot);
    if (lane == 0) {
      const int partial_offset = local_source * kMaxWarps + warp;
      reduction[partial_offset] = ssq;
      reduction[local_capacity * kMaxWarps + partial_offset] = query_dot;
      reduction[2 * local_capacity * kMaxWarps + partial_offset] = grad_dot;
    }
  }
  __syncthreads();

  for (int local_source = warp;
       local_source < local_sources;
       local_source += kMaxWarps) {
    const int source_base = (local_source / kMaxWarps) * kMaxWarps;
    const int wave_sources = min(kMaxWarps, local_sources - source_base);
    const int source_offset = local_source - source_base;
    const bool contributes = lane < kMaxWarps && lane % wave_sources == source_offset;
    const int partial_offset = local_source * kMaxWarps + lane;
    float ssq = contributes ? reduction[partial_offset] : 0.0f;
    float query_dot = contributes
        ? reduction[local_capacity * kMaxWarps + partial_offset]
        : 0.0f;
    float grad_dot = contributes
        ? reduction[2 * local_capacity * kMaxWarps + partial_offset]
        : 0.0f;
    warp_sum3(ssq, query_dot, grad_dot);
    if (lane == 0) {
      stats[kQueryDot * local_capacity + local_source] = query_dot;
      stats[kGradDot * local_capacity + local_source] = grad_dot;
      stats[kRstd * local_capacity + local_source] = rsqrtf(ssq / width + eps);
    }
  }

  cluster_sync<ClusterBlocks>();

  // One thread computes the source softmax and its backward scalars.
  if (rank == 0 && threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int source = 0; source < n_sources; ++source) {
      const int owner = source / local_capacity;
      const int local_source = source - owner * local_capacity;
      float* owner_stats = map_stats<ClusterBlocks>(stats, owner);
      const float logit = owner_stats[kQueryDot * local_capacity + local_source] *
          owner_stats[kRstd * local_capacity + local_source];
      maximum = fmaxf(maximum, logit);
    }

    float denominator = 0.0f;
    for (int source = 0; source < n_sources; ++source) {
      const int owner = source / local_capacity;
      const int local_source = source - owner * local_capacity;
      float* owner_stats = map_stats<ClusterBlocks>(stats, owner);
      const float logit = owner_stats[kQueryDot * local_capacity + local_source] *
          owner_stats[kRstd * local_capacity + local_source];
      denominator += expf(logit - maximum);
    }

    float centered_grad = 0.0f;
    for (int source = 0; source < n_sources; ++source) {
      const int owner = source / local_capacity;
      const int local_source = source - owner * local_capacity;
      float* owner_stats = map_stats<ClusterBlocks>(stats, owner);
      const float logit = owner_stats[kQueryDot * local_capacity + local_source] *
          owner_stats[kRstd * local_capacity + local_source];
      const float alpha = expf(logit - maximum) / denominator;
      owner_stats[kAlpha * local_capacity + local_source] = alpha;
      centered_grad += alpha * owner_stats[kGradDot * local_capacity + local_source];
    }

    for (int source = 0; source < n_sources; ++source) {
      const int owner = source / local_capacity;
      const int local_source = source - owner * local_capacity;
      float* owner_stats = map_stats<ClusterBlocks>(stats, owner);
      const float alpha = owner_stats[kAlpha * local_capacity + local_source];
      owner_stats[kDlogit * local_capacity + local_source] =
          alpha * (owner_stats[kGradDot * local_capacity + local_source] - centered_grad);
    }
  }

  cluster_sync<ClusterBlocks>();

  // Each thread owns a disjoint set of features, so every grad_values element
  // has exactly one writer.
  for (int feature = threadIdx.x; feature < width; feature += blockDim.x) {
    const float query_value = to_float(query[feature]);
    const float output_grad = to_float(shared_grad[feature]);
    float query_grad = 0.0f;
    for (int local_source = 0; local_source < local_sources; ++local_source) {
      const int source = source_begin + local_source;
      const float value = to_float(
          shared_values[static_cast<int64_t>(local_source) * width + feature]);
      const float alpha = stats[kAlpha * local_capacity + local_source];
      const float dlogit = stats[kDlogit * local_capacity + local_source];
      const float rstd = stats[kRstd * local_capacity + local_source];
      const float query_dot = stats[kQueryDot * local_capacity + local_source];
      const float da = dlogit * rstd;
      const float dssq = -dlogit * query_dot * rstd * rstd * rstd / (2.0f * width);
      const float value_grad = alpha * output_grad + da * query_value + 2.0f * dssq * value;
      const int64_t value_offset =
          static_cast<int64_t>(source) * stride_source +
          static_cast<int64_t>(token) * stride_token +
          static_cast<int64_t>(feature) * stride_feature;
      grad_values[value_offset] = from_float<scalar_t>(value_grad);
      query_grad += da * value;
    }
    if (local_sources > 0) {
      // This traffic-control path matches Liger's one contribution per token.
      atomicAdd(grad_query + feature, query_grad);
    }
  }
}

template <typename scalar_t>
__global__ void shared_backward_kernel(
    const scalar_t* values,
    const scalar_t* query,
    const scalar_t* grad_out,
    scalar_t* grad_values,
    float* grad_query,
    int n_sources,
    int n_tokens,
    int width,
    float eps,
    int64_t stride_source,
    int64_t stride_token,
    int64_t stride_feature) {
  shared_backward_body<scalar_t, 1>(
      values,
      query,
      grad_out,
      grad_values,
      grad_query,
      n_sources,
      n_tokens,
      width,
      eps,
      stride_source,
      stride_token,
      stride_feature);
}

// These production-oriented candidates shard D, not N, across two or four
// cluster blocks. Each source element, output-gradient element, and source-gradient
// element then has one global reader/writer. Exact forward coefficients remove
// every backward reduction except g dot v. The cluster remains persistent over
// a token subset and accumulates dw in shared memory, reducing contended global
// atomics from one per token to one per persistent cluster.
template <typename scalar_t, int ClusterBlocks>
__global__ __cluster_dims__(ClusterBlocks, 1, 1) void feature_cluster_backward_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ saved_alpha,
    const float* __restrict__ saved_rstd,
    const float* __restrict__ saved_norm,
    scalar_t* __restrict__ grad_values,
    float* __restrict__ grad_query,
    int n_sources,
    int n_tokens,
    int width,
    int64_t stride_source,
    int64_t stride_token,
    int64_t stride_feature) {
  extern __shared__ __align__(16) unsigned char storage[];
  const auto cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id = static_cast<int>(blockIdx.x) / ClusterBlocks;
  const int cluster_count = static_cast<int>(gridDim.x) / ClusterBlocks;
  const int feature_capacity = (width + ClusterBlocks - 1) / ClusterBlocks;
  const int feature_begin = rank * feature_capacity;
  const int local_width = max(0, min(feature_capacity, width - feature_begin));

  scalar_t* shared_values = reinterpret_cast<scalar_t*>(storage);
  const size_t value_bytes =
      static_cast<size_t>(n_sources) * feature_capacity * sizeof(scalar_t);
  scalar_t* shared_grad = shared_values +
      static_cast<int64_t>(n_sources) * feature_capacity;
  const size_t grad_bytes = static_cast<size_t>(feature_capacity) * sizeof(scalar_t);
  const size_t float_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  float* shared_dw = reinterpret_cast<float*>(storage + float_offset);
  float* stats = shared_dw + feature_capacity;
  float* grad_dot_partials = stats + kFeatureClusterFields * n_sources;

  for (int local_feature = threadIdx.x;
       local_feature < feature_capacity;
       local_feature += blockDim.x) {
    shared_dw[local_feature] = 0.0f;
  }
  __syncthreads();

  for (int token = cluster_id; token < n_tokens; token += cluster_count) {
    for (int local_feature = threadIdx.x;
         local_feature < local_width;
         local_feature += blockDim.x) {
      const int feature = feature_begin + local_feature;
      shared_grad[local_feature] =
          grad_out[static_cast<int64_t>(token) * width + feature];
    }
    __syncthreads();

    // Spread each source wave across every warp. A partial final wave assigns
    // several warps to each remaining source, which keeps N=9 and N=17 balanced.
    // Each warp writes a private g-dot-v partial. One barrier makes all partials
    // visible, and independent warps finish the source reductions. The former
    // implementation used two block barriers for every source.
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    for (int source_base = 0; source_base < n_sources; source_base += kMaxWarps) {
      const int wave_sources = min(kMaxWarps, n_sources - source_base);
      const int source_offset = warp % wave_sources;
      const int source_warp = warp / wave_sources;
      const int source_warps = 1 + (kMaxWarps - 1 - source_offset) / wave_sources;
      const int source = source_base + source_offset;
      float grad_dot = 0.0f;
      for (int local_feature = source_warp * kWarpSize + lane;
           local_feature < local_width;
           local_feature += source_warps * kWarpSize) {
        const int feature = feature_begin + local_feature;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token +
            static_cast<int64_t>(feature) * stride_feature;
        const scalar_t raw = values[value_offset];
        shared_values[static_cast<int64_t>(source) * feature_capacity + local_feature] = raw;
        grad_dot += to_float(raw) * to_float(shared_grad[local_feature]);
      }
      grad_dot = warp_sum(grad_dot);
      if (lane == 0) {
        grad_dot_partials[source * kMaxWarps + warp] = grad_dot;
      }
    }
    __syncthreads();

    for (int source = warp; source < n_sources; source += kMaxWarps) {
      const int source_base = (source / kMaxWarps) * kMaxWarps;
      const int wave_sources = min(kMaxWarps, n_sources - source_base);
      const int source_offset = source - source_base;
      float grad_dot =
          lane < kMaxWarps && lane % wave_sources == source_offset
          ? grad_dot_partials[source * kMaxWarps + lane]
          : 0.0f;
      grad_dot = warp_sum(grad_dot);
      if (lane == 0) {
        stats[source] = grad_dot;
      }
    }

    cluster.sync();

    // Rank zero's first warp combines all source scalars in parallel. One lane
    // owns each source, sums its feature-shard partials through DSM, and then
    // participates in the source-softmax dot product. This removes the former
    // serial O(N * cluster_blocks) control section.
    if (rank == 0 && warp == 0) {
      const int source = lane;
      float grad_dot = 0.0f;
      float alpha = 0.0f;
      if (source < n_sources) {
#pragma unroll
        for (int owner = 0; owner < ClusterBlocks; ++owner) {
          grad_dot += cluster.map_shared_rank(stats, owner)[source];
        }
        const int64_t saved_offset =
            static_cast<int64_t>(source) * n_tokens + token;
        alpha = saved_alpha[saved_offset];
      }
      float centered_grad = warp_sum(alpha * grad_dot);
      centered_grad = __shfl_sync(0xffffffffu, centered_grad, 0);
      if (source < n_sources) {
        const int64_t saved_offset =
            static_cast<int64_t>(source) * n_tokens + token;
        const float rstd = saved_rstd[saved_offset];
        const float norm = saved_norm[saved_offset];
        const float dlogit = alpha * (grad_dot - centered_grad);
#pragma unroll
        for (int owner = 0; owner < ClusterBlocks; ++owner) {
          float* owner_stats = cluster.map_shared_rank(stats, owner);
          owner_stats[n_sources + source] = dlogit;
          owner_stats[2 * n_sources + source] = alpha;
          owner_stats[3 * n_sources + source] = rstd;
          owner_stats[4 * n_sources + source] = norm;
        }
      }
    }

    cluster.sync();

    for (int local_feature = threadIdx.x;
         local_feature < local_width;
         local_feature += blockDim.x) {
      const int feature = feature_begin + local_feature;
      const float output_grad = to_float(shared_grad[local_feature]);
      const float query_value = to_float(query[feature]);
      float query_grad = 0.0f;
      for (int source = 0; source < n_sources; ++source) {
        const float value = to_float(
            shared_values[static_cast<int64_t>(source) * feature_capacity + local_feature]);
        const float alpha = stats[2 * n_sources + source];
        const float rstd = stats[3 * n_sources + source];
        const float norm = stats[4 * n_sources + source];
        const float dlogit = stats[n_sources + source];
        const float value_grad =
            alpha * output_grad + dlogit * rstd * query_value - dlogit * norm * value;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token +
            static_cast<int64_t>(feature) * stride_feature;
        grad_values[value_offset] = from_float<scalar_t>(value_grad);
        query_grad += dlogit * rstd * value;
      }
      shared_dw[local_feature] += query_grad;
    }

    // The second cluster barrier completed every remote shared-memory access.
    // Only this block reads its local tile afterwards, so a block barrier is
    // sufficient before the same storage is reused for the next token.
    __syncthreads();
  }

  for (int local_feature = threadIdx.x;
       local_feature < local_width;
       local_feature += blockDim.x) {
    atomicAdd(grad_query + feature_begin + local_feature, shared_dw[local_feature]);
  }
}

// Fixed-shape persistent CTA. The anchor widths are multiples of 1024, so each
// thread owns the same feature pairs in every source. Raw low-precision pairs
// remain packed in registers between g-dot-v and gradient application. This
// reaches the one-read traffic floor without a source-sized shared-memory tile,
// DSM, or cluster barriers.
template <typename scalar_t, int NSources, int Width>
__global__ __launch_bounds__(kRegisterThreads, 1) void register_backward_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ saved_alpha,
    const float* __restrict__ saved_rstd,
    const float* __restrict__ saved_norm,
    scalar_t* __restrict__ grad_values,
    float* __restrict__ grad_query,
    int n_tokens,
    int64_t stride_source,
    int64_t stride_token) {
  static_assert(Width % 2 == 0);
  constexpr int kPairsPerSource = Width / 2;
  static_assert(kPairsPerSource % kRegisterThreads == 0);
  constexpr int kFeaturePairsPerThread = kPairsPerSource / kRegisterThreads;
  constexpr int kHeldPairs = NSources * kFeaturePairsPerThread;

  extern __shared__ __align__(16) float shared[];
  float* warp_partials = shared;
  float* stats = warp_partials + NSources * kRegisterWarps;
  float* grad_dots = stats;
  float* alphas = grad_dots + NSources;
  float* rstds = alphas + NSources;
  float* norms = rstds + NSources;
  float* dlogits = norms + NSources;
  float* shared_dw = dlogits + NSources;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  uint32_t held[kHeldPairs];
  uint32_t held_grad[kFeaturePairsPerThread];

  for (int feature = threadIdx.x; feature < Width; feature += blockDim.x) {
    shared_dw[feature] = 0.0f;
  }
  __syncthreads();

  for (int token = blockIdx.x; token < n_tokens; token += gridDim.x) {
#pragma unroll
    for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
      const int pair = threadIdx.x + slot * kRegisterThreads;
      held_grad[slot] = reinterpret_cast<const uint32_t*>(
          grad_out + static_cast<int64_t>(token) * Width + 2 * pair)[0];
    }
#pragma unroll
    for (int source = 0; source < NSources; ++source) {
      float grad_dot = 0.0f;
#pragma unroll
      for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
        const int pair = threadIdx.x + slot * kRegisterThreads;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token + 2 * pair;
        const uint32_t raw =
            reinterpret_cast<const uint32_t*>(values + value_offset)[0];
        held[source * kFeaturePairsPerThread + slot] = raw;
        const float2 value = PairOps<scalar_t>::unpack(raw);
        const float2 output_grad = PairOps<scalar_t>::unpack(held_grad[slot]);
        grad_dot += value.x * output_grad.x + value.y * output_grad.y;
      }
      grad_dot = warp_sum(grad_dot);
      if (lane == 0) {
        warp_partials[source * kRegisterWarps + warp] = grad_dot;
      }
    }
    __syncthreads();

    if (warp == 0) {
      float centered_grad = 0.0f;
#pragma unroll
      for (int source = 0; source < NSources; ++source) {
        float grad_dot =
            lane < kRegisterWarps
            ? warp_partials[source * kRegisterWarps + lane]
            : 0.0f;
        grad_dot = warp_sum(grad_dot);
        if (lane == 0) {
          const int64_t saved_offset =
              static_cast<int64_t>(source) * n_tokens + token;
          const float alpha = saved_alpha[saved_offset];
          grad_dots[source] = grad_dot;
          alphas[source] = alpha;
          rstds[source] = saved_rstd[saved_offset];
          norms[source] = saved_norm[saved_offset];
          centered_grad += alpha * grad_dot;
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int source = 0; source < NSources; ++source) {
          dlogits[source] = alphas[source] * (grad_dots[source] - centered_grad);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
      const int pair = threadIdx.x + slot * kRegisterThreads;
      const int feature = 2 * pair;
      const float2 query_value = PairOps<scalar_t>::unpack(
          reinterpret_cast<const uint32_t*>(query + feature)[0]);
      const float2 output_grad = PairOps<scalar_t>::unpack(held_grad[slot]);
      float dw_x = 0.0f;
      float dw_y = 0.0f;
#pragma unroll
      for (int source = 0; source < NSources; ++source) {
        const float2 value = PairOps<scalar_t>::unpack(
            held[source * kFeaturePairsPerThread + slot]);
        const float alpha = alphas[source];
        const float dlogit = dlogits[source];
        const float rstd = rstds[source];
        const float norm = norms[source];
        const float dv_x = alpha * output_grad.x +
            dlogit * rstd * query_value.x - dlogit * norm * value.x;
        const float dv_y = alpha * output_grad.y +
            dlogit * rstd * query_value.y - dlogit * norm * value.y;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token + feature;
        reinterpret_cast<uint32_t*>(grad_values + value_offset)[0] =
            PairOps<scalar_t>::pack(dv_x, dv_y);
        dw_x += dlogit * rstd * value.x;
        dw_y += dlogit * rstd * value.y;
      }
      shared_dw[feature] += dw_x;
      shared_dw[feature + 1] += dw_y;
    }
  }

#pragma unroll
  for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
    const int feature = 2 * (threadIdx.x + slot * kRegisterThreads);
    atomicAdd(grad_query + feature, shared_dw[feature]);
    atomicAdd(grad_query + feature + 1, shared_dw[feature + 1]);
  }
}

// Two 512-thread blocks shard D while retaining their source pairs in
// registers. This extends the spill-free register design to the wide and
// many-source gap shapes. Only per-source scalar reductions cross DSM.
template <typename scalar_t, int NSources, int Width, int ClusterBlocks>
__global__ __cluster_dims__(ClusterBlocks, 1, 1) __launch_bounds__(kRegisterThreads, 1)
void register_cluster_backward_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ saved_alpha,
    const float* __restrict__ saved_rstd,
    const float* __restrict__ saved_norm,
    scalar_t* __restrict__ grad_values,
    float* __restrict__ grad_query,
    int n_tokens,
    int64_t stride_source,
    int64_t stride_token) {
  constexpr int kLocalWidth = Width / ClusterBlocks;
  static_assert(Width % ClusterBlocks == 0);
  static_assert(kLocalWidth % 2 == 0);
  constexpr int kPairsPerBlock = kLocalWidth / 2;
  static_assert(kPairsPerBlock % kRegisterThreads == 0);
  constexpr int kFeaturePairsPerThread = kPairsPerBlock / kRegisterThreads;
  constexpr int kHeldPairs = NSources * kFeaturePairsPerThread;

  extern __shared__ __align__(16) float shared[];
  float* warp_partials = shared;
  float* stats = warp_partials + NSources * kRegisterWarps;
  float* grad_dots = stats;
  float* alphas = grad_dots + NSources;
  float* rstds = alphas + NSources;
  float* norms = rstds + NSources;
  float* dlogits = norms + NSources;
  float* shared_dw = dlogits + NSources;
  volatile uint32_t* shared_grad = reinterpret_cast<uint32_t*>(
      shared_dw + kLocalWidth);

  const auto cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id = static_cast<int>(blockIdx.x) / ClusterBlocks;
  const int cluster_count = static_cast<int>(gridDim.x) / ClusterBlocks;
  const int feature_begin = rank * kLocalWidth;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  uint32_t held[kHeldPairs];

  for (int local_feature = threadIdx.x;
       local_feature < kLocalWidth;
       local_feature += blockDim.x) {
    shared_dw[local_feature] = 0.0f;
  }
  __syncthreads();

  for (int token = cluster_id; token < n_tokens; token += cluster_count) {
#pragma unroll
    for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
      const int local_pair = threadIdx.x + slot * kRegisterThreads;
      const int feature = feature_begin + 2 * local_pair;
      shared_grad[local_pair] = reinterpret_cast<const uint32_t*>(
          grad_out + static_cast<int64_t>(token) * Width + feature)[0];
    }

#pragma unroll
    for (int source = 0; source < NSources; ++source) {
      float grad_dot = 0.0f;
#pragma unroll
      for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
        const int local_pair = threadIdx.x + slot * kRegisterThreads;
        const int feature = feature_begin + 2 * local_pair;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token + feature;
        const uint32_t raw = reinterpret_cast<const uint32_t*>(values + value_offset)[0];
        held[source * kFeaturePairsPerThread + slot] = raw;
        const float2 value = PairOps<scalar_t>::unpack(raw);
        const float2 output_grad = PairOps<scalar_t>::unpack(shared_grad[local_pair]);
        grad_dot += value.x * output_grad.x + value.y * output_grad.y;
      }
      grad_dot = warp_sum(grad_dot);
      if (lane == 0) {
        warp_partials[source * kRegisterWarps + warp] = grad_dot;
      }
    }
    __syncthreads();

    for (int source = warp; source < NSources; source += kRegisterWarps) {
      float grad_dot = lane < kRegisterWarps
          ? warp_partials[source * kRegisterWarps + lane]
          : 0.0f;
      grad_dot = warp_sum(grad_dot);
      if (lane == 0) {
        grad_dots[source] = grad_dot;
      }
    }
    cluster.sync();

    if (rank == 0 && warp == 0) {
      const int source = lane;
      float grad_dot = 0.0f;
      float alpha = 0.0f;
      if (source < NSources) {
#pragma unroll
        for (int owner = 0; owner < ClusterBlocks; ++owner) {
          grad_dot += cluster.map_shared_rank(grad_dots, owner)[source];
        }
        const int64_t saved_offset =
            static_cast<int64_t>(source) * n_tokens + token;
        alpha = saved_alpha[saved_offset];
      }
      float centered_grad = warp_sum(alpha * grad_dot);
      centered_grad = __shfl_sync(0xffffffffu, centered_grad, 0);
      if (source < NSources) {
        const int64_t saved_offset =
            static_cast<int64_t>(source) * n_tokens + token;
        const float rstd = saved_rstd[saved_offset];
        const float norm = saved_norm[saved_offset];
        const float dlogit = alpha * (grad_dot - centered_grad);
#pragma unroll
        for (int owner = 0; owner < ClusterBlocks; ++owner) {
          float* owner_stats = cluster.map_shared_rank(stats, owner);
          owner_stats[NSources + source] = alpha;
          owner_stats[2 * NSources + source] = rstd;
          owner_stats[3 * NSources + source] = norm;
          owner_stats[4 * NSources + source] = dlogit;
        }
      }
    }
    cluster.sync();

#pragma unroll
    for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
      const int local_pair = threadIdx.x + slot * kRegisterThreads;
      const int local_feature = 2 * local_pair;
      const int feature = feature_begin + local_feature;
      const float2 query_value = PairOps<scalar_t>::unpack(
          reinterpret_cast<const uint32_t*>(query + feature)[0]);
      const float2 output_grad = PairOps<scalar_t>::unpack(shared_grad[local_pair]);
      float dw_x = 0.0f;
      float dw_y = 0.0f;
#pragma unroll
      for (int source = 0; source < NSources; ++source) {
        const float2 value = PairOps<scalar_t>::unpack(
            held[source * kFeaturePairsPerThread + slot]);
        const float alpha = alphas[source];
        const float rstd = rstds[source];
        const float norm = norms[source];
        const float dlogit = dlogits[source];
        const float dv_x = alpha * output_grad.x +
            dlogit * rstd * query_value.x - dlogit * norm * value.x;
        const float dv_y = alpha * output_grad.y +
            dlogit * rstd * query_value.y - dlogit * norm * value.y;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token + feature;
        reinterpret_cast<uint32_t*>(grad_values + value_offset)[0] =
            PairOps<scalar_t>::pack(dv_x, dv_y);
        dw_x += dlogit * rstd * value.x;
        dw_y += dlogit * rstd * value.y;
      }
      shared_dw[local_feature] += dw_x;
      shared_dw[local_feature + 1] += dw_y;
    }
    __syncthreads();
  }

#pragma unroll
  for (int slot = 0; slot < kFeaturePairsPerThread; ++slot) {
    const int local_feature =
        2 * (threadIdx.x + slot * kRegisterThreads);
    const int feature = feature_begin + local_feature;
    atomicAdd(grad_query + feature, shared_dw[local_feature]);
    atomicAdd(grad_query + feature + 1, shared_dw[local_feature + 1]);
  }
}

template <typename scalar_t>
size_t shared_bytes(int n_sources, int width, int cluster_blocks) {
  const int local_capacity = (n_sources + cluster_blocks - 1) / cluster_blocks;
  const size_t value_bytes = static_cast<size_t>(local_capacity) * width * sizeof(scalar_t);
  const size_t grad_bytes = static_cast<size_t>(width) * sizeof(scalar_t);
  const size_t stats_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  return stats_offset + sizeof(float) *
      (kStatFields * local_capacity + 3 * local_capacity * kMaxWarps);
}

template <typename scalar_t>
size_t feature_cluster_shared_bytes(int n_sources, int width, int cluster_blocks) {
  const int feature_capacity = (width + cluster_blocks - 1) / cluster_blocks;
  const size_t value_bytes =
      static_cast<size_t>(n_sources) * feature_capacity * sizeof(scalar_t);
  const size_t grad_bytes = static_cast<size_t>(feature_capacity) * sizeof(scalar_t);
  const size_t float_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  return float_offset + sizeof(float) *
      (feature_capacity + (kFeatureClusterFields + kMaxWarps) * n_sources);
}

template <int NSources, int Width>
constexpr size_t register_shared_bytes() {
  return sizeof(float) * (NSources * (kRegisterWarps + kStatFields) + Width);
}

template <typename scalar_t, int NSources, int Width, int ClusterBlocks>
constexpr size_t register_cluster_shared_bytes() {
  return sizeof(float) *
      (NSources * (kRegisterWarps + kStatFields) + Width / ClusterBlocks) +
      sizeof(scalar_t) * (Width / ClusterBlocks);
}

template <typename scalar_t, int NSources, int Width>
std::vector<int64_t> register_occupancy(int device) {
  constexpr size_t dynamic_shared = register_shared_bytes<NSources, Width>();
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, register_backward_kernel<scalar_t, NSources, Width>));
  int active_blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      register_backward_kernel<scalar_t, NSources, Width>,
      kRegisterThreads,
      dynamic_shared));
  int multiprocessors = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device));
  return {
      static_cast<int64_t>(active_blocks_per_sm) * multiprocessors,
      static_cast<int64_t>(dynamic_shared),
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(multiprocessors),
      kRegisterThreads,
  };
}

template <typename scalar_t, int NSources, int Width, int ClusterBlocks>
std::vector<int64_t> register_cluster_occupancy(int device) {
  constexpr size_t dynamic_shared =
      register_cluster_shared_bytes<scalar_t, NSources, Width, ClusterBlocks>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      register_cluster_backward_kernel<scalar_t, NSources, Width, ClusterBlocks>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_shared)));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes,
      register_cluster_backward_kernel<scalar_t, NSources, Width, ClusterBlocks>));
  int multiprocessors = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device));
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(multiprocessors * ClusterBlocks, 1, 1);
  config.blockDim = dim3(kRegisterThreads, 1, 1);
  config.dynamicSmemBytes = dynamic_shared;
  config.stream = at::cuda::getCurrentCUDAStream(device);
  int active_clusters = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &active_clusters,
      register_cluster_backward_kernel<scalar_t, NSources, Width, ClusterBlocks>,
      &config));
  return {
      static_cast<int64_t>(active_clusters),
      static_cast<int64_t>(dynamic_shared),
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(multiprocessors),
      kRegisterThreads,
      ClusterBlocks,
  };
}

template <typename scalar_t, int NSources, int Width>
void launch_register_specialization(
    const torch::Tensor& values,
    const torch::Tensor& query,
    const torch::Tensor& grad_out,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_rstd,
    const torch::Tensor& saved_norm,
    torch::Tensor& grad_values,
    torch::Tensor& grad_query) {
  const int n_tokens = static_cast<int>(values.size(1) * values.size(2));
  const auto occupancy = register_occupancy<scalar_t, NSources, Width>(
      values.get_device());
  const int persistent_blocks = std::min(n_tokens, static_cast<int>(occupancy[0]));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device());
  register_backward_kernel<scalar_t, NSources, Width>
      <<<persistent_blocks,
         kRegisterThreads,
         register_shared_bytes<NSources, Width>(),
         stream>>>(
          reinterpret_cast<const scalar_t*>(values.data_ptr()),
          reinterpret_cast<const scalar_t*>(query.data_ptr()),
          reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
          saved_alpha.data_ptr<float>(),
          saved_rstd.data_ptr<float>(),
          saved_norm.data_ptr<float>(),
          reinterpret_cast<scalar_t*>(grad_values.data_ptr()),
          grad_query.data_ptr<float>(),
          n_tokens,
          values.stride(0),
          values.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int NSources, int Width, int ClusterBlocks>
void launch_register_cluster_specialization(
    const torch::Tensor& values,
    const torch::Tensor& query,
    const torch::Tensor& grad_out,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_rstd,
    const torch::Tensor& saved_norm,
    torch::Tensor& grad_values,
    torch::Tensor& grad_query) {
  const int n_tokens = static_cast<int>(values.size(1) * values.size(2));
  const auto occupancy =
      register_cluster_occupancy<scalar_t, NSources, Width, ClusterBlocks>(
          values.get_device());
  const int persistent_clusters = std::min(n_tokens, static_cast<int>(occupancy[0]));
  TORCH_CHECK(persistent_clusters > 0, "register cluster has zero achievable occupancy");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device());
  register_cluster_backward_kernel<scalar_t, NSources, Width, ClusterBlocks>
      <<<persistent_clusters * ClusterBlocks,
         kRegisterThreads,
         register_cluster_shared_bytes<scalar_t, NSources, Width, ClusterBlocks>(),
         stream>>>(
          reinterpret_cast<const scalar_t*>(values.data_ptr()),
          reinterpret_cast<const scalar_t*>(query.data_ptr()),
          reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
          saved_alpha.data_ptr<float>(),
          saved_rstd.data_ptr<float>(),
          saved_norm.data_ptr<float>(),
          reinterpret_cast<scalar_t*>(grad_values.data_ptr()),
          grad_query.data_ptr<float>(),
          n_tokens,
          values.stride(0),
          values.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_register_backward(
    const torch::Tensor& values,
    const torch::Tensor& query,
    const torch::Tensor& grad_out,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_rstd,
    const torch::Tensor& saved_norm,
    torch::Tensor& grad_values,
    torch::Tensor& grad_query) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  if (n_sources == 9 && width == 4096) {
    launch_register_specialization<scalar_t, 9, 4096>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  } else {
    TORCH_CHECK(
        false,
        "register backward supports only (N,D)=(9,4096)");
  }
}

template <typename scalar_t>
std::vector<int64_t> dispatch_register_occupancy(const torch::Tensor& values) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  if (n_sources == 9 && width == 4096) {
    return register_occupancy<scalar_t, 9, 4096>(values.get_device());
  }
  TORCH_CHECK(
      false,
      "register backward supports only (N,D)=(9,4096)");
}

template <typename scalar_t>
void dispatch_register_cluster_backward(
    const torch::Tensor& values,
    const torch::Tensor& query,
    const torch::Tensor& grad_out,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_rstd,
    const torch::Tensor& saved_norm,
    torch::Tensor& grad_values,
    torch::Tensor& grad_query) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  if (n_sources == 9 && width == 8192) {
    launch_register_cluster_specialization<scalar_t, 9, 8192, 4>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  } else if (n_sources == 32 && width == 2048) {
    launch_register_cluster_specialization<scalar_t, 32, 2048, 2>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  } else {
    TORCH_CHECK(
        false,
        "register cluster supports only (N,D)=(9,8192) or (32,2048)");
  }
}

template <typename scalar_t>
std::vector<int64_t> dispatch_register_cluster_occupancy(
    const torch::Tensor& values) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  if (n_sources == 9 && width == 8192) {
    return register_cluster_occupancy<scalar_t, 9, 8192, 4>(values.get_device());
  }
  if (n_sources == 32 && width == 2048) {
    return register_cluster_occupancy<scalar_t, 32, 2048, 2>(values.get_device());
  }
  TORCH_CHECK(
      false,
      "register cluster supports only (N,D)=(9,8192) or (32,2048)");
}

template <typename scalar_t>
void launch_backward(
    const torch::Tensor& values,
    const torch::Tensor& query,
    const torch::Tensor& grad_out,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_rstd,
    const torch::Tensor& saved_norm,
    torch::Tensor& grad_values,
    torch::Tensor& grad_query,
    float eps,
    int cluster_blocks) {
  const int n_sources = static_cast<int>(values.size(0));
  const int n_tokens = static_cast<int>(values.size(1) * values.size(2));
  const int width = static_cast<int>(values.size(3));
  const bool clustered = cluster_blocks > 1;
  TORCH_CHECK(
      cluster_blocks == 1 || cluster_blocks == 2 || cluster_blocks == 4,
      "cluster_blocks must be 1, 2, or 4");
  const size_t dynamic_shared = clustered
      ? feature_cluster_shared_bytes<scalar_t>(n_sources, width, cluster_blocks)
      : shared_bytes<scalar_t>(n_sources, width, cluster_blocks);

  int max_shared = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      values.get_device()));
  cudaFuncAttributes attributes{};
  if (cluster_blocks == 2) {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, feature_cluster_backward_kernel<scalar_t, 2>));
  } else if (cluster_blocks == 4) {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, feature_cluster_backward_kernel<scalar_t, 4>));
  } else {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, shared_backward_kernel<scalar_t>));
  }
  const size_t total_shared = dynamic_shared + attributes.sharedSizeBytes;
  TORCH_CHECK(
      total_shared <= static_cast<size_t>(max_shared),
      "shared backward needs ",
      dynamic_shared,
      " dynamic + ",
      attributes.sharedSizeBytes,
      " static shared bytes per block, but this device permits ",
      max_shared);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device());
  if (clustered) {
    int cluster_launch = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &cluster_launch,
        cudaDevAttrClusterLaunch,
        values.get_device()));
    TORCH_CHECK(cluster_launch, "device does not support thread-block clusters");
    TORCH_CHECK(n_sources <= 2 * kMaxLocalSources, "cluster path supports at most 32 sources");
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors,
        cudaDevAttrMultiProcessorCount,
        values.get_device()));
    cudaLaunchConfig_t occupancy_config{};
    occupancy_config.gridDim = dim3(multiprocessors * cluster_blocks, 1, 1);
    occupancy_config.blockDim = dim3(kThreads, 1, 1);
    occupancy_config.dynamicSmemBytes = dynamic_shared;
    occupancy_config.stream = stream;
    int active_clusters = 0;
    if (cluster_blocks == 2) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          feature_cluster_backward_kernel<scalar_t, 2>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(dynamic_shared)));
      C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
          &active_clusters,
          feature_cluster_backward_kernel<scalar_t, 2>,
          &occupancy_config));
    } else {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          feature_cluster_backward_kernel<scalar_t, 4>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(dynamic_shared)));
      C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
          &active_clusters,
          feature_cluster_backward_kernel<scalar_t, 4>,
          &occupancy_config));
    }
    TORCH_CHECK(active_clusters > 0, "feature-sharded cluster has zero achievable occupancy");
    const int persistent_clusters = std::min(n_tokens, active_clusters);
    if (cluster_blocks == 2) {
      feature_cluster_backward_kernel<scalar_t, 2>
          <<<persistent_clusters * 2, kThreads, dynamic_shared, stream>>>(
            reinterpret_cast<const scalar_t*>(values.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
            saved_alpha.data_ptr<float>(),
            saved_rstd.data_ptr<float>(),
            saved_norm.data_ptr<float>(),
            reinterpret_cast<scalar_t*>(grad_values.data_ptr()),
            grad_query.data_ptr<float>(),
            n_sources,
            n_tokens,
            width,
            values.stride(0),
            values.stride(2),
            values.stride(3));
    } else {
      feature_cluster_backward_kernel<scalar_t, 4>
          <<<persistent_clusters * 4, kThreads, dynamic_shared, stream>>>(
              reinterpret_cast<const scalar_t*>(values.data_ptr()),
              reinterpret_cast<const scalar_t*>(query.data_ptr()),
              reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
              saved_alpha.data_ptr<float>(),
              saved_rstd.data_ptr<float>(),
              saved_norm.data_ptr<float>(),
              reinterpret_cast<scalar_t*>(grad_values.data_ptr()),
              grad_query.data_ptr<float>(),
              n_sources,
              n_tokens,
              width,
              values.stride(0),
              values.stride(2),
              values.stride(3));
    }
  } else {
    TORCH_CHECK(n_sources <= kMaxLocalSources, "single-block path supports at most 16 sources");
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        shared_backward_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_shared)));
    shared_backward_kernel<scalar_t><<<n_tokens, kThreads, dynamic_shared, stream>>>(
        reinterpret_cast<const scalar_t*>(values.data_ptr()),
        reinterpret_cast<const scalar_t*>(query.data_ptr()),
        reinterpret_cast<const scalar_t*>(grad_out.data_ptr()),
        reinterpret_cast<scalar_t*>(grad_values.data_ptr()),
        grad_query.data_ptr<float>(),
        n_sources,
        n_tokens,
        width,
        eps,
        values.stride(0),
        values.stride(2),
        values.stride(3));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> shared_backward(
    torch::Tensor values,
    torch::Tensor query,
    torch::Tensor grad_out,
    torch::Tensor saved_alpha,
    torch::Tensor saved_rstd,
    torch::Tensor saved_norm,
    double eps,
    int cluster_blocks) {
  TORCH_CHECK(values.is_cuda() && query.is_cuda() && grad_out.is_cuda(), "all tensors must be CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.device() == query.device() && values.device() == grad_out.device(), "all tensors must share one device");
  TORCH_CHECK(values.is_contiguous() && query.is_contiguous() && grad_out.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(values.dim() == 4, "values must be [N, B, T, D]");
  TORCH_CHECK(query.dim() == 1 && query.size(0) == values.size(3), "query must be [D]");
  TORCH_CHECK(
      grad_out.sizes() == torch::IntArrayRef({values.size(1), values.size(2), values.size(3)}),
      "grad_out must be [B, T, D]");
  TORCH_CHECK(values.scalar_type() == query.scalar_type() && values.scalar_type() == grad_out.scalar_type(), "all tensors must have one dtype");
  TORCH_CHECK(
      values.scalar_type() == torch::kFloat16 || values.scalar_type() == torch::kBFloat16,
      "shared backward supports float16 and bfloat16");
  TORCH_CHECK(values.size(0) > 0 && values.size(1) > 0 && values.size(2) > 0 && values.size(3) > 0, "all dimensions must be positive");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(
      values.size(0) <= kIntMax && values.size(3) <= kIntMax &&
          values.size(1) <= kIntMax / values.size(2),
      "N, B*T, and D must fit in signed 32-bit kernel indices");
  TORCH_CHECK(eps > 0.0 && std::isfinite(eps), "eps must be finite and positive");

  int compute_major = 0;
  int compute_minor = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "shared backward was compiled for sm_120 but received sm_",
      compute_major,
      compute_minor);
  TORCH_CHECK(
      cluster_blocks == 1 || cluster_blocks == 2 || cluster_blocks == 4,
      "cluster_blocks must be 1, 2, or 4");
  if (cluster_blocks > 1) {
    const int64_t n_tokens = values.size(1) * values.size(2);
    const std::vector<int64_t> expected{values.size(0), n_tokens};
    for (const auto& saved : {saved_alpha, saved_rstd, saved_norm}) {
      TORCH_CHECK(saved.is_cuda() && saved.device() == values.device(), "saved state must use the values device");
      TORCH_CHECK(saved.scalar_type() == torch::kFloat32, "saved state must be fp32");
      TORCH_CHECK(saved.is_contiguous() && saved.sizes() == expected, "saved state must be contiguous [N, B*T]");
    }
  }

  auto grad_values = torch::empty_like(values);
  auto grad_query = torch::zeros(
      {values.size(3)},
      values.options().dtype(torch::kFloat32));

  if (values.scalar_type() == torch::kFloat16) {
    launch_backward<__half>(values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query, static_cast<float>(eps), cluster_blocks);
  } else {
    launch_backward<__nv_bfloat16>(values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query, static_cast<float>(eps), cluster_blocks);
  }
  return {grad_values, grad_query};
}

std::vector<torch::Tensor> register_backward(
    torch::Tensor values,
    torch::Tensor query,
    torch::Tensor grad_out,
    torch::Tensor saved_alpha,
    torch::Tensor saved_rstd,
    torch::Tensor saved_norm) {
  TORCH_CHECK(values.is_cuda() && query.is_cuda() && grad_out.is_cuda(), "all tensors must be CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.device() == query.device() && values.device() == grad_out.device(), "all tensors must share one device");
  TORCH_CHECK(values.is_contiguous() && query.is_contiguous() && grad_out.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(values.dim() == 4, "values must be [N, B, T, D]");
  TORCH_CHECK(query.dim() == 1 && query.size(0) == values.size(3), "query must be [D]");
  TORCH_CHECK(
      grad_out.sizes() == torch::IntArrayRef({values.size(1), values.size(2), values.size(3)}),
      "grad_out must be [B, T, D]");
  TORCH_CHECK(values.scalar_type() == query.scalar_type() && values.scalar_type() == grad_out.scalar_type(), "all tensors must have one dtype");
  TORCH_CHECK(
      values.scalar_type() == torch::kFloat16 || values.scalar_type() == torch::kBFloat16,
      "register backward supports float16 and bfloat16");
  TORCH_CHECK(values.size(0) > 0 && values.size(1) > 0 && values.size(2) > 0 && values.size(3) > 0, "all dimensions must be positive");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(
      values.size(0) <= kIntMax && values.size(3) <= kIntMax &&
          values.size(1) <= kIntMax / values.size(2),
      "N, B*T, and D must fit in signed 32-bit kernel indices");
  const int64_t n_tokens = values.size(1) * values.size(2);
  const std::vector<int64_t> expected{values.size(0), n_tokens};
  for (const auto& saved : {saved_alpha, saved_rstd, saved_norm}) {
    TORCH_CHECK(saved.is_cuda() && saved.device() == values.device(), "saved state must use the values device");
    TORCH_CHECK(saved.scalar_type() == torch::kFloat32, "saved state must be fp32");
    TORCH_CHECK(saved.is_contiguous() && saved.sizes() == expected, "saved state must be contiguous [N, B*T]");
  }
  int compute_major = 0;
  int compute_minor = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "register backward was compiled for sm_120 but received sm_",
      compute_major,
      compute_minor);

  auto grad_values = torch::empty_like(values);
  auto grad_query = torch::zeros(
      {values.size(3)}, values.options().dtype(torch::kFloat32));
  if (values.scalar_type() == torch::kFloat16) {
    dispatch_register_backward<__half>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  } else {
    dispatch_register_backward<__nv_bfloat16>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  }
  return {grad_values, grad_query};
}

std::vector<torch::Tensor> register_cluster_backward(
    torch::Tensor values,
    torch::Tensor query,
    torch::Tensor grad_out,
    torch::Tensor saved_alpha,
    torch::Tensor saved_rstd,
    torch::Tensor saved_norm) {
  TORCH_CHECK(values.is_cuda() && query.is_cuda() && grad_out.is_cuda(), "all tensors must be CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.device() == query.device() && values.device() == grad_out.device(), "all tensors must share one device");
  TORCH_CHECK(values.is_contiguous() && query.is_contiguous() && grad_out.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(values.dim() == 4, "values must be [N, B, T, D]");
  TORCH_CHECK(query.dim() == 1 && query.size(0) == values.size(3), "query must be [D]");
  TORCH_CHECK(
      grad_out.sizes() == torch::IntArrayRef({values.size(1), values.size(2), values.size(3)}),
      "grad_out must be [B, T, D]");
  TORCH_CHECK(values.scalar_type() == query.scalar_type() && values.scalar_type() == grad_out.scalar_type(), "all tensors must have one dtype");
  TORCH_CHECK(
      values.scalar_type() == torch::kFloat16 || values.scalar_type() == torch::kBFloat16,
      "register cluster supports float16 and bfloat16");
  TORCH_CHECK(values.size(0) > 0 && values.size(1) > 0 && values.size(2) > 0 && values.size(3) > 0, "all dimensions must be positive");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(
      values.size(0) <= kIntMax && values.size(3) <= kIntMax &&
          values.size(1) <= kIntMax / values.size(2),
      "N, B*T, and D must fit in signed 32-bit kernel indices");
  const int64_t n_tokens = values.size(1) * values.size(2);
  const std::vector<int64_t> expected{values.size(0), n_tokens};
  for (const auto& saved : {saved_alpha, saved_rstd, saved_norm}) {
    TORCH_CHECK(saved.is_cuda() && saved.device() == values.device(), "saved state must use the values device");
    TORCH_CHECK(saved.scalar_type() == torch::kFloat32, "saved state must be fp32");
    TORCH_CHECK(saved.is_contiguous() && saved.sizes() == expected, "saved state must be contiguous [N, B*T]");
  }
  int compute_major = 0;
  int compute_minor = 0;
  int cluster_launch = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cluster_launch, cudaDevAttrClusterLaunch, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "register cluster was compiled for sm_120 but received sm_",
      compute_major,
      compute_minor);
  TORCH_CHECK(cluster_launch, "device does not support thread-block clusters");

  auto grad_values = torch::empty_like(values);
  auto grad_query = torch::zeros(
      {values.size(3)}, values.options().dtype(torch::kFloat32));
  if (values.scalar_type() == torch::kFloat16) {
    dispatch_register_cluster_backward<__half>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  } else {
    dispatch_register_cluster_backward<__nv_bfloat16>(
        values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query);
  }
  return {grad_values, grad_query};
}

template <typename scalar_t, int ClusterBlocks>
std::vector<int64_t> cluster_launch_info_t(const torch::Tensor& values) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  const size_t dynamic_shared =
      feature_cluster_shared_bytes<scalar_t>(n_sources, width, ClusterBlocks);
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, feature_cluster_backward_kernel<scalar_t, ClusterBlocks>));
  int max_shared = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      values.get_device()));
  TORCH_CHECK(
      dynamic_shared + attributes.sharedSizeBytes <= static_cast<size_t>(max_shared),
      "feature-sharded cluster exceeds the per-block shared-memory limit");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      feature_cluster_backward_kernel<scalar_t, ClusterBlocks>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_shared)));
  int multiprocessors = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessors,
      cudaDevAttrMultiProcessorCount,
      values.get_device()));
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(multiprocessors * ClusterBlocks, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = dynamic_shared;
  config.stream = at::cuda::getCurrentCUDAStream(values.get_device());
  int active_clusters = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &active_clusters,
      feature_cluster_backward_kernel<scalar_t, ClusterBlocks>,
      &config));
  return {
      static_cast<int64_t>(active_clusters),
      static_cast<int64_t>(dynamic_shared),
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(max_shared),
      static_cast<int64_t>(multiprocessors),
      kThreads,
      ClusterBlocks,
  };
}

std::vector<int64_t> cluster_launch_info(torch::Tensor values, int cluster_blocks) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 4, "values must be a CUDA [N, B, T, D] tensor");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(values.size(0) > 0 && values.size(0) <= 32, "cluster supports 1 to 32 sources");
  TORCH_CHECK(values.size(1) > 0 && values.size(2) > 0 && values.size(3) > 0, "all dimensions must be positive");
  TORCH_CHECK(cluster_blocks == 2 || cluster_blocks == 4, "cluster_blocks must be 2 or 4");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(
      values.size(3) <= kIntMax && values.size(1) <= kIntMax / values.size(2),
      "B*T and D must fit in signed 32-bit kernel indices");
  int compute_major = 0;
  int compute_minor = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "cluster launch information requires sm_120, received sm_",
      compute_major,
      compute_minor);
  if (values.scalar_type() == torch::kFloat16) {
    return cluster_blocks == 2
        ? cluster_launch_info_t<__half, 2>(values)
        : cluster_launch_info_t<__half, 4>(values);
  }
  TORCH_CHECK(values.scalar_type() == torch::kBFloat16, "cluster supports float16 and bfloat16");
  return cluster_blocks == 2
      ? cluster_launch_info_t<__nv_bfloat16, 2>(values)
      : cluster_launch_info_t<__nv_bfloat16, 4>(values);
}

std::vector<int64_t> register_launch_info(torch::Tensor values) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 4, "values must be a CUDA [N, B, T, D] tensor");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(values.size(1) > 0 && values.size(2) > 0, "B and T must be positive");
  int compute_major = 0;
  int compute_minor = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "register launch information requires sm_120, received sm_",
      compute_major,
      compute_minor);
  if (values.scalar_type() == torch::kFloat16) {
    return dispatch_register_occupancy<__half>(values);
  }
  TORCH_CHECK(values.scalar_type() == torch::kBFloat16, "register path supports float16 and bfloat16");
  return dispatch_register_occupancy<__nv_bfloat16>(values);
}

std::vector<int64_t> register_cluster_launch_info(torch::Tensor values) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 4, "values must be a CUDA [N, B, T, D] tensor");
  const c10::cuda::CUDAGuard device_guard(values.device());
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(values.size(1) > 0 && values.size(2) > 0, "B and T must be positive");
  int compute_major = 0;
  int compute_minor = 0;
  int cluster_launch = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_major, cudaDevAttrComputeCapabilityMajor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &compute_minor, cudaDevAttrComputeCapabilityMinor, values.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cluster_launch, cudaDevAttrClusterLaunch, values.get_device()));
  TORCH_CHECK(
      compute_major == 12 && compute_minor == 0,
      "register cluster launch information requires sm_120, received sm_",
      compute_major,
      compute_minor);
  TORCH_CHECK(cluster_launch, "device does not support thread-block clusters");
  if (values.scalar_type() == torch::kFloat16) {
    return dispatch_register_cluster_occupancy<__half>(values);
  }
  TORCH_CHECK(values.scalar_type() == torch::kBFloat16, "register cluster supports float16 and bfloat16");
  return dispatch_register_cluster_occupancy<__nv_bfloat16>(values);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "shared_backward",
      &shared_backward,
      "Block AttnRes shared-memory backward",
      pybind11::arg("values"),
      pybind11::arg("query"),
      pybind11::arg("grad_out"),
      pybind11::arg("saved_alpha"),
      pybind11::arg("saved_rstd"),
      pybind11::arg("saved_norm"),
      pybind11::arg("eps"),
      pybind11::arg("cluster_blocks"));
  module.def(
      "cluster_launch_info",
      &cluster_launch_info,
      "Block AttnRes feature-cluster launch information",
      pybind11::arg("values"),
      pybind11::arg("cluster_blocks"));
  module.def(
      "register_backward",
      &register_backward,
      "Block AttnRes register-resident backward",
      pybind11::arg("values"),
      pybind11::arg("query"),
      pybind11::arg("grad_out"),
      pybind11::arg("saved_alpha"),
      pybind11::arg("saved_rstd"),
      pybind11::arg("saved_norm"));
  module.def(
      "register_launch_info",
      &register_launch_info,
      "Block AttnRes register-resident launch information",
      pybind11::arg("values"));
  module.def(
      "register_cluster_backward",
      &register_cluster_backward,
      "Block AttnRes register-cluster backward",
      pybind11::arg("values"),
      pybind11::arg("query"),
      pybind11::arg("grad_out"),
      pybind11::arg("saved_alpha"),
      pybind11::arg("saved_rstd"),
      pybind11::arg("saved_norm"));
  module.def(
      "register_cluster_launch_info",
      &register_cluster_launch_info,
      "Block AttnRes register-cluster launch information",
      pybind11::arg("values"));
}
