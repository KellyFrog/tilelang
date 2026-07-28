/*!
 * \file tl/config.h
 * \brief TileLang configuration utilities.
 */

#ifndef TVM_TL_CONFIG_H_
#define TVM_TL_CONFIG_H_

#include <tvm/ffi/optional.h>
#include <tvm/ir/transform.h>

#include "op/builtin.h"

namespace tvm {
namespace tl {
namespace tl_config {

/*!
 * \brief Check if vectorize planner verbose output is enabled.
 */
inline bool VectorizePlannerVerboseEnabled() {
  auto ctxt = ::tvm::transform::PassContext::Current();
  return ctxt
      ->GetConfig("tl.enable_vectorize_planner_verbose", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

/*!
 * \brief Check if 256-bit vectorization is disabled.
 */
inline bool Vectorize256Disabled() {
  auto ctxt = ::tvm::transform::PassContext::Current();
  return ctxt->GetConfig("tl.disable_vectorize_256", ffi::Optional<Bool>())
      .value_or(Bool(false));
}

/*!
 * \brief First named-barrier (bar.sync) ID handed out to reductions.
 *
 * Barrier ID 0 is reserved for __syncthreads. The default 1 keeps the first
 * AllReduce in a kernel on barrier ID 1, preserving the legacy codegen for
 * single-reduction kernels. Subsequent AllReduces rotate through 2, 3, ... so
 * multiple reductions never collide.
 */
inline int NamedBarrierStart() {
  auto ctxt = ::tvm::transform::PassContext::Current();
  int start = static_cast<int>(
      ctxt->GetConfig(kNamedBarrierStart, ffi::Optional<Integer>())
          .value_or(Integer(1))
          ->value);
  // Hardware provides named barriers 1..15 (0 is __syncthreads).
  if (start < 1 || start > 15) {
    LOG(FATAL) << "tl.named_barrier_start must be in [1, 15], got " << start;
  }
  return start;
}

} // namespace tl_config
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CONFIG_H_
