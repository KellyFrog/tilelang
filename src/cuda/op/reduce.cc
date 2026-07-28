/*!
 * \file tl/cuda/op/reduce.cc
 * \brief CUDA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace cuda {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsCuda(target);
  }

  static int GetPreferedVectorizedSize(DataType dt, Target target) {
    if (!TargetIsCuda(target)) {
      return 1;
    }
    bool supports_fp32x2 = TargetHasSMVersionGE(target, 100);
    return backend::reduce::GetPreferedVectorizedSize(dt, supports_fp32x2);
  }

  static std::string
  MakeBatchAllReduce(std::string reducer, int reducing_threads, int scale,
                     PrimExpr thread_offset, PrimExpr all_threads, int batch,
                     int workspace_stride, Target target,
                     const backend::reduce::AllReduceBarrier &barrier) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ", " << barrier.barrier_id
         << ">, " << batch << ", " << workspace_stride;
    } else {
      ss << ", tl::SyncThreadsBarrier, " << batch << ", " << workspace_stride;
    }
    ss << ">::run_batch";
    return ss.str();
  }

  static std::string
  MakeScalarAllReduce(std::string reducer, int reducing_threads, int scale,
                      PrimExpr thread_offset, PrimExpr all_threads,
                      Target target,
                      const backend::reduce::AllReduceBarrier &barrier) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (barrier.participants > 0) {
      // Partial-CTA reduction: exactly `barrier.participants` threads, a
      // contiguous range starting at `thread_offset`, arrive at the
      // barrier. `thread_offset` doubles as the shared-memory base so the
      // butterfly exchange stays inside the participating range.
      ss << ", tl::NamedBarrier<" << barrier.participants << ", "
         << barrier.barrier_id << ">";
    } else if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ", " << barrier.barrier_id
         << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cuda.Reduce",
      MatchCudaReduceTarget,
      cuda::Reduce::Lower,
  });
  return true;
}

const bool cuda_reduce_registered = RegisterCudaReduce();

} // namespace

} // namespace tl
} // namespace tvm
