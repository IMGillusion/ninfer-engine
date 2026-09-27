# Phase-1 host KV prewrite candidate

Baseline snapshot commit: 18c556c1 (current E:/ninfer-src tracked + untracked source snapshot). This worktree never changes production source files or containers.

Enable only for testing: `NINFER_HOST_KV_PREWRITE=1`; unset or any other value is OFF. No CLI/config budget changes. Disk 64GiB, host 6GiB and state 16 unchanged.

After pressure demotion publishes, register private-owner main/backend full pages. Shared-prefix owners, partial tails, recurrent state, active eviction and checkpoint guarantees are explicitly outside this first phase. Full queue refuses registration and increments `full`, not `written`; owner/page stays resident, existing owner-death fallback remains intact.

Engine worker boundary executes under execution_mutex, after cancellation. Tick defers during context transactions/pending transactions/state forks. Idle worker uses timed wake only while background work remains. Candidate validation checks catalogued owner generation, live address mapping and exact logical handle, content epoch, host residency, no writer, full committed page and digest availability/equality. Reuse/drop invalidates the candidate. Copies allocate ordinary owned memory before short pin/memcpy/unpin; no global scratch, state slot or GPU transfer is used.

Limits: fixed 4096 candidate descriptors; at most 32 invalid inspections, one page copy and one admission per tick, 8MiB page/buffer cap and 250us cooperative time quantum. Allocation/memcpy is non-preemptible, so 250us is NOT a hard wall-clock latency bound; overrun postpones admission to the next boundary. Actual scheduler delay needs measurement. One page in flight; bridge releases owned capacity before completion publication. Bridge Busy retains bytes+candidate and revalidates next time, without copy repetition or queue-drop success accounting. An accepted owned write remains valid after owner death, and worker never dereferences owner memory. A failed write is counted failed; resident bytes remain available for normal fallback. Completion means readable upsert, not durable index or a whole recoverable checkpoint.

Reproduce CPU checks with read-only bind mount:
`docker run --rm --network none --entrypoint bash -v C:/Users/ZGQ/ninfer-host-prewrite:/src:ro -w /src vllm/vllm-openai:nightly /src/tests/run_host_prewrite.sh`

Outstanding: real GPU demote->register->idle/active tick integration, real owner reuse/drop and content-epoch mutation under pressure, cancellation/fairness measurements, state-backed end-to-end restore, TSAN and full linked-image verification. No image build or deployment performed. Unit generation invalidation uses the shared scheduler with a mock owner validator; full runtime validation is syntax checked, not GPU exercised.
