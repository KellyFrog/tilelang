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
 * \brief First auto-allocated (non-reduce) named-barrier (bar.sync) ID.
 *
 * Reductions claim barrier IDs cycling through [1, K-1] (one ID per reduction,
 * reused across butterfly phases as generations), and auto-allocated barriers
 * (ThreadSync shared-memory syncs) start at K. Barrier ID 0 is reserved for
 * __syncthreads and hardware provides IDs 0..15, so K is in [2, 15]. The
 * default 3 keeps barrier IDs 1 and 2 for reductions.
 */
inline int NamedBarrierStart() {
  auto ctxt = ::tvm::transform::PassContext::Current();
  int start = static_cast<int>(
      ctxt->GetConfig(kNamedBarrierStart, ffi::Optional<Integer>())
          .value_or(Integer(3))
          ->value);
  if (start < 2 || start > 15) {
    LOG(FATAL) << "tl.named_barrier_start must be in [2, 15], got " << start;
  }
  return start;
}

} // namespace tl_config
} // namespace tl
} // namespace tvm

#endif // TVM_TL_CONFIG_H_
