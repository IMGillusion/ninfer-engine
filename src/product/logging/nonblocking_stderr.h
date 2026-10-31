#pragma once
// Non-blocking stderr resilience. The engine writes its operational log to
// stderr, which inside a container is a pipe consumed by the runtime's log
// driver. When that consumer dies (observed: dockerd died while the engine
// kept running), the pipe fills and every fprintf(stderr) blocks - the
// writing thread hangs holding whatever lock it held, the stall sentinel
// dies inside its own forensic log call before reaching _Exit, and the
// whole engine wedges (observed: 5 hours, all 52 threads in futex_wait).
//
// This layer swaps fd 2 for an O_NONBLOCK pipe early in main: business
// threads' writes either land in the pipe or fail with EAGAIN (the line is
// dropped; fprintf's return value is already ignored everywhere) - they can
// never block. A single drain thread moves bytes from the pipe to the real
// stderr (fd saved at install time); if that write wedges, only the drain
// thread is lost and the engine keeps running with logs dropped.
//
// Must be installed before any thread that writes logs starts. Idempotent.

namespace ninfer::product {

// Swap fd 2 for a non-blocking pipe and start the drain thread. Returns
// false if installation failed (the process keeps its original stderr; no
// worse than before). Call once from main before spawning threads.
bool install_nonblocking_stderr();

} // namespace ninfer::product
