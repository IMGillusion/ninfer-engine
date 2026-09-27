# Cooperative disk seed integration status

## Implemented in this worktree (not deployed)
- Transaction owns staged Prefill and SeedProgress. Root disk restore is prepared once while continuation remains ReservedMaterialization and lane remains Empty. Only start_request activates/publishes the lane; finalization moves Prefill into RequestControl.
- Real transaction progress submits/polls typed CPU read tickets and returns InProgress to the engine. State is read first; KV is read/copied one page per scheduler visit. No synchronous disk restore remains in the seed path.
- StateImageDevicePool has a byte-cursor chunk API. Each visit submits at most one region and 4 MiB. Test uses 37-byte chunks to exercise region crossings, destination slot offset and distinct payload bytes.
- H2D is deliberately synchronous on device.stream, ordered after mapping/reset on that same stream. No independent stream or pipeline transfer-event wait introduced. Synchronization/driver contention can still overrun the byte-based scheduling bound.
- Cancellation is sticky; accepted read drains before any abort releases reserved sequence/scratch. Exceptional teardown waits for borrowed CPU reads. The singleton context transaction serializes owner spill and seed access to startup scratch.
- Missing/corrupt/read-admission error yields advertised_base=0 and ordinary full recompute. Full prompt service reservation remains unchanged. No disk identity is dropped because of a transfer/read failure.
- Shared SeedLifecycle is used by actual Program prepare/final publication/cancellation cleanup and unit-tested for duplicate prepare/finalize, invalid publication and pending-read cancellation.

## Verified
See seed-verified.log and scripts/verify_seed.sh for reproducible commands.
- Both exact 27b and 35b_a3b variant translation units syntax compile.
- Actual CPU bridge/store read, batch-probe, asynchronous tickets, owner-batch bridge, owner-batch scheduler and owner plan regressions.
- SeedLifecycle + actual bridge tickets PASS; ASAN/UBSAN PASS.
- Actual StateImageDevicePool GPU H2D chunk and D2H readback PASS (not a simulated copy).

## Explicit remaining gates / merge warnings
- No end-to-end Program/model request ran. Lifecycle helper tests exercise production guard logic, NOT real Program initialization/cancel/error state transitions; that requested acceptance remains open.
- Need isolated model request tests: valid disk hit, missing later page->full recompute completion, subsequent request, slow read while another lane decodes, cancel during read and between H2D chunks. No scheduler latency claim yet.
- CUDA_CHECK still inherits the existing abort behavior. Separate CUDA recovery patch must be integrated/audited for partial submission/drain before production. This candidate does not claim GPU-error recovery.
- State chunk API is now bounded, but each cudaStreamSynchronize can wait on prior compute-stream work. Validate pipeline dependency ordering with real concurrent decode before calling it production-ready.
- No conditional corruption invalidation added; corrupt identities are retained rather than incorrectly deleting a concurrently repaired identity.
- Full image build tag ninfer:l3-seedyield1 uses this worktree only; inspect seed-image-build.log and recorded process status. No production container touched, no commit/push.
