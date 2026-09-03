#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kThreads / kWarpSize;
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

__device__ __forceinline__ void block_sum3(
    float& x,
    float& y,
    float& z,
    float* scratch) {
  constexpr unsigned kFullMask = 0xffffffffu;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    x += __shfl_down_sync(kFullMask, x, offset);
    y += __shfl_down_sync(kFullMask, y, offset);
    z += __shfl_down_sync(kFullMask, z, offset);
  }
  if (lane == 0) {
    scratch[warp] = x;
    scratch[kMaxWarps + warp] = y;
    scratch[2 * kMaxWarps + warp] = z;
  }
  __syncthreads();

  if (warp == 0) {
    x = lane < kMaxWarps ? scratch[lane] : 0.0f;
    y = lane < kMaxWarps ? scratch[kMaxWarps + lane] : 0.0f;
    z = lane < kMaxWarps ? scratch[2 * kMaxWarps + lane] : 0.0f;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      x += __shfl_down_sync(kFullMask, x, offset);
      y += __shfl_down_sync(kFullMask, y, offset);
      z += __shfl_down_sync(kFullMask, z, offset);
    }
    if (lane == 0) {
      scratch[0] = x;
      scratch[1] = y;
      scratch[2] = z;
    }
  }
  __syncthreads();
  x = scratch[0];
  y = scratch[1];
  z = scratch[2];
}

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

  // Read each source element from global memory exactly once. The raw low-
  // precision value remains in shared memory until both gradient equations
  // have consumed it.
  for (int local_source = 0; local_source < local_sources; ++local_source) {
    const int source = source_begin + local_source;
    float ssq = 0.0f;
    float query_dot = 0.0f;
    float grad_dot = 0.0f;
    for (int feature = threadIdx.x; feature < width; feature += blockDim.x) {
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
    block_sum3(ssq, query_dot, grad_dot, reduction);
    if (threadIdx.x == 0) {
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

// The production-oriented candidate shards D, not N, across two cluster
// blocks. Each source element, output-gradient element, and source-gradient
// element then has one global reader/writer. Exact forward coefficients remove
// every backward reduction except g dot v. The cluster remains persistent over
// a token subset and accumulates dw in shared memory, reducing contended global
// atomics from one per token to one per persistent cluster.
template <typename scalar_t>
__global__ __cluster_dims__(2, 1, 1) void feature_cluster_backward_kernel(
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
  constexpr int kClusterBlocks = 2;

  const auto cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id = static_cast<int>(blockIdx.x) / kClusterBlocks;
  const int cluster_count = static_cast<int>(gridDim.x) / kClusterBlocks;
  const int feature_capacity = (width + kClusterBlocks - 1) / kClusterBlocks;
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
  float* reduction = stats + kFeatureClusterFields * n_sources;

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

    // Load every source element exactly once and retain it until dv and dw are
    // complete. Each block computes the g dot v contribution for its D shard.
    for (int source = 0; source < n_sources; ++source) {
      float grad_dot = 0.0f;
      for (int local_feature = threadIdx.x;
           local_feature < local_width;
           local_feature += blockDim.x) {
        const int feature = feature_begin + local_feature;
        const int64_t value_offset =
            static_cast<int64_t>(source) * stride_source +
            static_cast<int64_t>(token) * stride_token +
            static_cast<int64_t>(feature) * stride_feature;
        const scalar_t raw = values[value_offset];
        shared_values[static_cast<int64_t>(source) * feature_capacity + local_feature] = raw;
        grad_dot += to_float(raw) * to_float(shared_grad[local_feature]);
      }
      float unused_y = 0.0f;
      float unused_z = 0.0f;
      block_sum3(grad_dot, unused_y, unused_z, reduction);
      if (threadIdx.x == 0) {
        stats[source] = grad_dot;
      }
    }

    cluster.sync();

    // One thread combines only N scalar reductions through distributed shared
    // memory. It writes dlogit into both blocks so the feature paths stay
    // independent after the second cluster barrier.
    if (rank == 0 && threadIdx.x == 0) {
      float* rank_one_stats = cluster.map_shared_rank(stats, 1);
      float centered_grad = 0.0f;
      for (int source = 0; source < n_sources; ++source) {
        const float grad_dot = stats[source] + rank_one_stats[source];
        const int64_t saved_offset = static_cast<int64_t>(source) * n_tokens + token;
        const float alpha = saved_alpha[saved_offset];
        stats[2 * n_sources + source] = alpha;
        rank_one_stats[2 * n_sources + source] = alpha;
        centered_grad += alpha * grad_dot;
      }
      for (int source = 0; source < n_sources; ++source) {
        const float grad_dot = stats[source] + rank_one_stats[source];
        const int64_t saved_offset = static_cast<int64_t>(source) * n_tokens + token;
        const float alpha = stats[2 * n_sources + source];
        const float dlogit = alpha * (grad_dot - centered_grad);
        stats[n_sources + source] = dlogit;
        rank_one_stats[n_sources + source] = dlogit;
        const float rstd = saved_rstd[saved_offset];
        const float norm = saved_norm[saved_offset];
        stats[3 * n_sources + source] = rstd;
        rank_one_stats[3 * n_sources + source] = rstd;
        stats[4 * n_sources + source] = norm;
        rank_one_stats[4 * n_sources + source] = norm;
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

    // No block may overwrite its retained source tile until the peer has
    // completed the cluster-wide scalar exchange and gradient application.
    cluster.sync();
  }

  for (int local_feature = threadIdx.x;
       local_feature < local_width;
       local_feature += blockDim.x) {
    atomicAdd(grad_query + feature_begin + local_feature, shared_dw[local_feature]);
  }
}

template <typename scalar_t>
size_t shared_bytes(int n_sources, int width, int cluster_blocks) {
  const int local_capacity = (n_sources + cluster_blocks - 1) / cluster_blocks;
  const size_t value_bytes = static_cast<size_t>(local_capacity) * width * sizeof(scalar_t);
  const size_t grad_bytes = static_cast<size_t>(width) * sizeof(scalar_t);
  const size_t stats_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  return stats_offset + sizeof(float) * (kStatFields * local_capacity + 3 * kMaxWarps);
}

template <typename scalar_t>
size_t feature_cluster_shared_bytes(int n_sources, int width) {
  const int feature_capacity = (width + 1) / 2;
  const size_t value_bytes =
      static_cast<size_t>(n_sources) * feature_capacity * sizeof(scalar_t);
  const size_t grad_bytes = static_cast<size_t>(feature_capacity) * sizeof(scalar_t);
  const size_t float_offset = (value_bytes + grad_bytes + 15u) & ~size_t{15u};
  return float_offset + sizeof(float) *
      (feature_capacity + kFeatureClusterFields * n_sources + 3 * kMaxWarps);
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
    bool clustered) {
  const int n_sources = static_cast<int>(values.size(0));
  const int n_tokens = static_cast<int>(values.size(1) * values.size(2));
  const int width = static_cast<int>(values.size(3));
  const int cluster_blocks = clustered ? 2 : 1;
  const size_t dynamic_shared = clustered
      ? feature_cluster_shared_bytes<scalar_t>(n_sources, width)
      : shared_bytes<scalar_t>(n_sources, width, cluster_blocks);

  int max_shared = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      values.get_device()));
  cudaFuncAttributes attributes{};
  if (clustered) {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, feature_cluster_backward_kernel<scalar_t>));
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
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        feature_cluster_backward_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_shared)));
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors,
        cudaDevAttrMultiProcessorCount,
        values.get_device()));
    cudaLaunchConfig_t occupancy_config{};
    occupancy_config.gridDim = dim3(multiprocessors * 2, 1, 1);
    occupancy_config.blockDim = dim3(kThreads, 1, 1);
    occupancy_config.dynamicSmemBytes = dynamic_shared;
    occupancy_config.stream = stream;
    int active_clusters = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
        &active_clusters,
        feature_cluster_backward_kernel<scalar_t>,
        &occupancy_config));
    TORCH_CHECK(active_clusters > 0, "feature-sharded cluster has zero achievable occupancy");
    const int persistent_clusters = std::min(n_tokens, active_clusters);
    feature_cluster_backward_kernel<scalar_t>
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
    bool clustered) {
  TORCH_CHECK(values.is_cuda() && query.is_cuda() && grad_out.is_cuda(), "all tensors must be CUDA tensors");
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
  TORCH_CHECK(eps > 0.0 && std::isfinite(eps), "eps must be finite and positive");
  if (clustered) {
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
    launch_backward<__half>(values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query, static_cast<float>(eps), clustered);
  } else {
    launch_backward<__nv_bfloat16>(values, query, grad_out, saved_alpha, saved_rstd, saved_norm, grad_values, grad_query, static_cast<float>(eps), clustered);
  }
  return {grad_values, grad_query};
}

template <typename scalar_t>
std::vector<int64_t> cluster_launch_info_t(const torch::Tensor& values) {
  const int n_sources = static_cast<int>(values.size(0));
  const int width = static_cast<int>(values.size(3));
  const size_t dynamic_shared = feature_cluster_shared_bytes<scalar_t>(n_sources, width);
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, feature_cluster_backward_kernel<scalar_t>));
  int max_shared = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      values.get_device()));
  TORCH_CHECK(
      dynamic_shared + attributes.sharedSizeBytes <= static_cast<size_t>(max_shared),
      "feature-sharded cluster exceeds the per-block shared-memory limit");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      feature_cluster_backward_kernel<scalar_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_shared)));
  int multiprocessors = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessors,
      cudaDevAttrMultiProcessorCount,
      values.get_device()));
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(multiprocessors * 2, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = dynamic_shared;
  config.stream = at::cuda::getCurrentCUDAStream(values.get_device());
  int active_clusters = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &active_clusters,
      feature_cluster_backward_kernel<scalar_t>,
      &config));
  return {
      static_cast<int64_t>(active_clusters),
      static_cast<int64_t>(dynamic_shared),
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(max_shared),
      static_cast<int64_t>(multiprocessors),
      kThreads,
  };
}

std::vector<int64_t> cluster_launch_info(torch::Tensor values) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 4, "values must be a CUDA [N, B, T, D] tensor");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(values.size(0) > 0 && values.size(0) <= 32, "cluster supports 1 to 32 sources");
  TORCH_CHECK(values.size(1) > 0 && values.size(2) > 0 && values.size(3) > 0, "all dimensions must be positive");
  if (values.scalar_type() == torch::kFloat16) {
    return cluster_launch_info_t<__half>(values);
  }
  TORCH_CHECK(values.scalar_type() == torch::kBFloat16, "cluster supports float16 and bfloat16");
  return cluster_launch_info_t<__nv_bfloat16>(values);
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
      pybind11::arg("clustered"));
  module.def(
      "cluster_launch_info",
      &cluster_launch_info,
      "Block AttnRes feature-cluster launch information",
      pybind11::arg("values"));
}
