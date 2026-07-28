/*!
 * \file tl/backend/common/op/reduce.cc
 * \brief Shared non-template tl.reduce lowering utilities.
 */

#include "reduce.h"

#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {
namespace reduce {

using namespace tirx;
using namespace ffi;

AllReduceBarrier ResolveAllReduceBarrier(const Fragment &red_layout,
                                         const Range &thread_bounds,
                                         const Target &target) {
  AllReduceBarrier barrier;
  const int64_t *block_min = as_const_int(thread_bounds->min);
  const int64_t *block_extent = as_const_int(thread_bounds->extent);
  const int64_t *replicate = as_const_int(red_layout->ReplicateExtent());
  if (block_min == nullptr || block_extent == nullptr || replicate == nullptr) {
    LOG(FATAL) << "tl.reduce: cannot resolve the partial scalar AllReduce "
                  "barrier: the CTA thread bounds or the reduce layout's "
                  "replicate extent are not compile-time constants.";
  }
  ICHECK_GT(*replicate, 0)
      << "tl.reduce: reduce layout replicate extent must be positive";

  arith::Analyzer analyzer;
  PrimExpr thread_expr = red_layout->GetForwardThread();
  // Bind every variable the thread expression depends on: layout input
  // placeholders get their input-shape ranges, and the replicate variable
  // (the only other variable a reduce layout's thread expression contains)
  // gets [0, ReplicateExtent).
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> placeholders;
  for (size_t i = 0; i < red_layout->InputShape().size(); ++i) {
    Var placeholder = InputPlaceholder(i);
    placeholders.insert(placeholder);
    analyzer.Bind(placeholder, Range::FromMinExtent(
                                   Integer(0), red_layout->InputShape()[i]));
  }
  PostOrderVisit(thread_expr, [&](const ObjectRef &node) {
    const auto *var = node.as<VarNode>();
    if (var == nullptr) {
      return;
    }
    Var ref = GetRef<Var>(var);
    if (!placeholders.count(ref)) {
      analyzer.Bind(ref, Range::FromMinExtent(Integer(0), Integer(*replicate)));
    }
  });
  auto bound = analyzer.const_int_bound(thread_expr);
  if (bound->min_value == arith::ConstIntBoundNode::kNegInf ||
      bound->max_value == arith::ConstIntBoundNode::kPosInf) {
    LOG(FATAL) << "tl.reduce: cannot resolve the partial scalar AllReduce "
                  "barrier: the reduce layout's thread image is not a "
                  "compile-time constant.";
  }

  // Enumerate the full thread image (every output-index x replicate
  // combination) to compute the exact participant set and verify it is one
  // contiguous range. The participant count is the image span, not
  // ReplicateExtent: that is only the per-output-element replication, and a
  // multi-output reduce layout spans replicate * num_outputs threads (e.g.
  // two adjacent 64-thread groups). This also verifies density for the
  // whole-CTA case before we decide between the whole-CTA and partial
  // barriers.
  std::vector<std::pair<Var, int64_t>> enum_vars;
  for (size_t i = 0; i < red_layout->InputShape().size(); ++i) {
    const int64_t *extent = as_const_int(red_layout->InputShape()[i]);
    if (extent == nullptr) {
      LOG(FATAL) << "tl.reduce: reduce layout input shape is not a "
                    "compile-time constant.";
    }
    enum_vars.emplace_back(InputPlaceholder(i), *extent);
  }
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> rep_vars;
  PostOrderVisit(thread_expr, [&](const ObjectRef &node) {
    const auto *var = node.as<VarNode>();
    if (var == nullptr) {
      return;
    }
    Var ref = GetRef<Var>(var);
    if (!placeholders.count(ref)) {
      rep_vars.insert(ref);
    }
  });
  for (const Var &var : rep_vars) {
    enum_vars.emplace_back(var, *replicate);
  }

  constexpr int64_t kMaxCombos = 1 << 20;
  int64_t total_combos = 1;
  for (const auto &[var, extent] : enum_vars) {
    if (extent <= 0 || extent > kMaxCombos / total_combos) {
      total_combos = -1;
      break;
    }
    total_combos *= extent;
  }
  if (total_combos <= 0) {
    LOG(FATAL) << "tl.reduce: reduce layout thread image too large to "
                  "statically resolve the AllReduce barrier.";
  }

  std::set<int64_t> image;
  Map<Var, PrimExpr> binds;
  std::function<void(size_t)> enumerate = [&](size_t idx) {
    if (idx == enum_vars.size()) {
      PrimExpr value = analyzer.Simplify(Substitute(thread_expr, binds));
      const int64_t *constant = as_const_int(value);
      if (constant == nullptr) {
        LOG(FATAL) << "tl.reduce: failed to statically evaluate thread image "
                      "value "
                   << value;
      }
      image.insert(*constant);
      return;
    }
    const auto &[var, extent] = enum_vars[idx];
    for (int64_t i = 0; i < extent; ++i) {
      binds.Set(var, Integer(i));
      enumerate(idx + 1);
    }
  };
  enumerate(0);
  if (image.empty()) {
    LOG(FATAL) << "tl.reduce: failed to statically resolve the partial "
                  "AllReduce barrier thread image.";
  }
  const int64_t image_min = *image.begin();
  const int64_t image_max = *image.rbegin();
  const int64_t image_count = static_cast<int64_t>(image.size());
  ICHECK_EQ(image_count, image_max - image_min + 1)
      << "tl.reduce: partial scalar AllReduce barrier requires one "
         "contiguous thread range, but the reduce layout's thread image is "
         "not contiguous ("
      << image_count << " distinct values spanning [" << image_min << ", "
      << image_max << "])";
  ICHECK_EQ(image_min, bound->min_value)
      << "tl.reduce: internal error resolving the partial AllReduce barrier "
         "thread image";
  ICHECK_EQ(image_max, bound->max_value)
      << "tl.reduce: internal error resolving the partial AllReduce barrier "
         "thread image";

  if (image_min == *block_min && image_count == *block_extent) {
    // Whole-CTA participation: PartitionLoop drops the guard, so the
    // whole-CTA barrier the backend emits by default is correct.
    return barrier;
  }

  int64_t warp_size = 32;
  if (auto warp_size_attr = target->GetAttr<Integer>("thread_warp_size")) {
    warp_size = warp_size_attr.value()->value;
  }
  ICHECK_EQ((image_min - *block_min) % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce barrier requires a "
         "warp-aligned participating thread range, got base "
      << image_min << " in a block starting at " << *block_min;
  ICHECK_EQ(image_count % warp_size, 0)
      << "tl.reduce: partial scalar AllReduce barrier requires a "
         "warp-aligned thread range, got "
      << image_count << " threads";
  ICHECK(image_min >= *block_min && image_max < *block_min + *block_extent)
      << "tl.reduce: partial scalar AllReduce participating thread range ["
      << image_min << ", " << image_max
      << "] must lie within "
         "the CTA thread bounds ["
      << *block_min << ", " << *block_min + *block_extent << ")";

  barrier.partial = true;
  barrier.base = image_min;
  barrier.participants = image_count;
  return barrier;
}

int ClaimNamedBarrier(const LowerArgs &lower_args) {
  if (lower_args.named_barrier_next_id == nullptr) {
    return 1;
  }
  int id = *lower_args.named_barrier_next_id;
  *lower_args.named_barrier_next_id = id + 1;
  int cycle = std::max(1, lower_args.named_barrier_cycle);
  return ((id - 1) % cycle) + 1;
}

} // namespace reduce
} // namespace backend
} // namespace tl
} // namespace tvm
