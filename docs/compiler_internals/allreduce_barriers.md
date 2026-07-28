# AllReduce Barrier Lowering

TileLang's CUDA AllReduce implementation uses a barrier between cross-warp
butterfly steps. The selected barrier policy depends on the target:

- **SM80 and later:** `tl::NamedBarrier<count, id>`, lowered to PTX
  `bar.sync id, count`.
- **Earlier CUDA architectures:** `tl::SyncThreadsBarrier`, lowered to
  `__syncthreads()`.
- **Single-warp reductions:** shuffle instructions only; no CTA barrier is
  emitted.

`bar.sync` named barriers are distinct from Hopper's shared-memory
`mbarrier` instructions. The AllReduce path only requires the former, so it is
enabled starting with Ampere rather than Hopper.

The pre-SM80 fallback is a whole-CTA synchronization path. Partial-CTA scalar
AllReduce therefore requires SM80 or later.

## Partial-CTA Participation

A scalar reduction may be executed by only part of a CTA. Emitting a whole-CTA
barrier inside that guarded region would deadlock, so lowering derives the
exact participating thread range from the reduce fragment layout.

The forward thread expression is converted to an absolute CTA thread ID by
adding the fragment's `ThreadRange.min`. The compiler then:

1. binds every logical layout dimension and the replication dimension;
2. computes the minimum and maximum thread IDs with the arithmetic analyzer;
3. uses Z3 model enumeration to count distinct thread IDs in the image; and
4. requires the count to equal `max - min + 1`.

The final equality proves that the image is one contiguous range instead of a
sparse set that merely has the same bounds. The range must also lie inside the
CTA and be warp aligned. Its minimum becomes the AllReduce workspace offset,
and its exact size becomes the named-barrier arrival count.

This analysis enumerates at most the CTA thread count, not the Cartesian
product of layout coordinates. Large logical dimensions that do not affect the
thread expression therefore do not increase the search space.

## Barrier ID Allocation

Barrier ID 0 is reserved for the `__syncthreads()` convention. Cross-warp
AllReduces claim IDs cyclically from `[1, tl.named_barrier_start - 1]`; the
default `tl.named_barrier_start = 3` reserves IDs 1 and 2 for reductions.
Compiler-generated non-reduction barriers start at `tl.named_barrier_start`.

One reduction reuses its ID across butterfly steps as successive hardware
barrier generations. Sequential reductions may therefore reuse an ID safely.
If up to `C` reductions on different thread groups can be live concurrently,
set `tl.named_barrier_start` to at least `C + 1`, subject to the supported range
`[2, 15]`.

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_NAMED_BARRIER_START: 4,
    }
)
def make_kernel():
    ...
```

This configuration gives reductions IDs 1, 2, and 3, while automatic
non-reduction barriers begin at ID 4.
