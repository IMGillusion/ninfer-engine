#pragma once
// Engine-facing facade over DiskKVStore (the L3 disk cold tier).
//
// The engine drives this with per-PAGE content identities and host byte
// spans. One page = one slot in the store.
//
// Identity model (what makes this tier work):
//   The engine maintains a rolling 128-bit digest per token frontier
//   (PrefixShortlistDigests): digest(F) is a pure function of the first F
//   tokens (with their positions). A KV page covering [p*64, F) is
//   content-addressed by digest(F) — recomputable at spill time (from the
//   live sequence) AND at restore time (from the incoming request's own
//   digest chain) with no persistent metadata. Same content in two sessions
//   => same digest => the second spill is an idempotent no-op (free dedupe
//   of shared prefixes).
//
//   `tag` disambiguates KV produced for different engine configurations
//   (speculative backend, proposal head, KV dtype) — identical token
//   prefixes under a different spec backend produce different KV bytes, so
//   the tag is part of the key, exactly like the engine's own
//   CheckpointSummary::shortlist_key identity_tag.
//
//   A miss (evicted, corrupt, or never spilt) returns false and the engine
//   falls back to exactly what it does today (recompute). Strictly
//   better-or-equal.
//
// One family per KV plane: MainKV / BackendKV have different page strides,
// so each owns its own store file.

#include "core/disk_kv_store.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {

enum class DiskKVKind : std::uint8_t {
    MainKV      = 0,
    BackendKV   = 1,
    StateImage  = 2,  // reserved: state images (GDN recurrent state)
};

/** One queued spill: identity + copied bytes. Engine copies; the bridge's
 *  worker thread does the disk write (off the engine's hot path). */
// Completion acknowledges upsert/readability, NOT durable index publication.
// Probe runs on the writer too, so dedupe never waits for the store lock on the engine.
enum class SpillStatus : std::uint8_t { Pending, Present, Missing, Written, Dropped, Failed };
struct SpillCompletion { std::atomic<SpillStatus> status{SpillStatus::Pending}; };
using SpillTicket = std::shared_ptr<SpillCompletion>;

// One bounded batch consumes ONE queue slot and ONE scheduler completion.
inline constexpr std::size_t kDiskKVProbeBatchMax = 32;
struct ProbeBatchCompletion {
    // Read results only after poll(ticket) returns true (acquire publication).
    std::array<SpillStatus, kDiskKVProbeBatchMax> results{};
    std::size_t size = 0;
    std::atomic<bool> ready{false};
};
using ProbeBatchTicket = std::shared_ptr<ProbeBatchCompletion>;

struct SpillJob {
    DiskKVIdentity id;
    DiskKVKind kind;
    std::vector<std::byte> bytes;
    SpillTicket completion;
    std::span<const std::byte> borrowed;
    bool probe = false;
    std::vector<DiskKVIdentity> probe_ids;
    ProbeBatchTicket probe_batch;
    std::shared_ptr<struct ReadCompletion> read;
    std::span<std::byte> read_dst;
    // Maintenance work (proactive preparation) must not queue behind a flood of
    // on-demand spills: without this the preparation waited ~1.7s per page and
    // never finished, which is exactly the blocking this path exists to remove.
    bool priority = false;
};

struct ReadCompletion {
    std::atomic<DiskReadResult> result{DiskReadResult::Pending};
};
using ReadTicket = std::shared_ptr<ReadCompletion>;

struct DiskKVBridgeStats {
    std::uint64_t spills          = 0;
    std::uint64_t spill_bytes     = 0;
    std::uint64_t spill_dups      = 0;  // identity already restorable (dedupe hit)
    std::uint64_t restores        = 0;
    std::uint64_t restore_bytes   = 0;
    std::uint64_t restore_misses  = 0;
    std::uint64_t evicted_slots   = 0;
    std::uint64_t queue_drops     = 0;  // queue full: page lost (recompute later)
};

class DiskKVBridge {
public:
    struct Options {
        std::string base_path = "";  // parent dir; per-family files <name>.diskkv
        std::size_t main_page_stride    = 0;  // from HostKVPageLayout(MainKV)
        std::size_t backend_page_stride = 0;  // 0 = family disabled
        std::size_t state_page_stride   = 0;  // 0 = family disabled
        // Total spill budget; split across ENABLED families 85/15 (main dominates).
        std::size_t capacity_bytes = 0;  // 0 = disabled
        bool verify_crc = true;
    };

    /** Open (create) the per-family backing files. Throws on failure. */
    explicit DiskKVBridge(Options opts);
    ~DiskKVBridge();

    DiskKVBridge(const DiskKVBridge&)            = delete;
    DiskKVBridge& operator=(const DiskKVBridge&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    struct ClosureLease {
        std::array<DiskKVStore*, 3> stores{};
        ~ClosureLease() { for (auto* store : stores) if (store) store->unprotect(); }
    };
    std::unique_ptr<ClosureLease> try_protect_closure(
        const std::array<std::vector<DiskKVIdentity>, 3>& ids) {
        auto lease = std::make_unique<ClosureLease>();
        for (unsigned i = 0; i < 3; ++i) {
            if (ids[i].empty()) continue;
            auto* store = family(static_cast<DiskKVKind>(i)).store.get();
            if (!store || ids[i].size() > store->slot_count()) return {};
            auto keys = std::make_shared<const DiskKVStore::ProtectedKeys>(ids[i].begin(), ids[i].end());
            if (!store->try_protect(std::move(keys))) return {};
            lease->stores[i] = store;
        }
        return lease;
    }

    /** Spill one page. The bytes are copied into a bounded queue and written
     *  by a background thread: the caller only pays one memcpy (host pages
     *  are released by the engine right after this call, so synchronous disk
     *  IO would stall the pressure/eviction path for seconds). Returns true
     *  if the page is queued (or already restorable); false if the family is
     *  disabled or the queue is full (restore will fall back to recompute). */
    bool spill_page(const DiskKVIdentity& id, DiskKVKind kind, std::span<const std::byte> bytes);

    /** Backpressure-aware spill for the owner-death path: waits up to
     *  `timeout` for a free queue slot instead of dropping the page. A dropped
     *  page breaks the contiguous prefix chain (restore walks from 0 and stops
     *  at the first miss), costing a full recompute. Returns true if queued
     *  (or already restorable); false on timeout / disabled / shutdown. */
    bool spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                         std::span<const std::byte> bytes, std::chrono::milliseconds timeout);

    /** Synchronous spill: write the bytes to the store on the calling thread
     *  (no async queue copy). For large blobs (state images ~146 MiB) where
     *  buffering a copy in the bounded queue would exhaust memory. The caller
     *  must ensure `bytes` outlives this call (it is read directly). Returns
     *  true if restorable after the call (fresh write or dedupe hit). */
    bool spill_page_sync(const DiskKVIdentity& id, DiskKVKind kind,
                         std::span<const std::byte> bytes);

    // Null ticket = busy; retry without releasing the owner. No store lock.
    // Borrowed bytes MUST remain immutable/alive until poll != Pending,
    // including cancellation. Worker never accesses SequenceState.
    SpillTicket try_probe(const DiskKVIdentity& id, DiskKVKind kind);
    /** Nonblocking queue admission; one homogeneous batch, 1..32 identities.
     *  Identities are COPIED before return; no borrowed owner/page memory.
     *  Null = busy/shutdown: retry the ENTIRE unchanged batch, no drops counted.
     *  Empty/oversized input throws invalid_argument; allocation may throw.
     *  Disabled bridge/family or invalid kind completes every page as Failed.
     *  Results preserve input order (including duplicates); read only after
     *  poll returns true. Present is a contains() snapshot, not a CRC check,
     *  reservation, cross-page atomic snapshot, or durable-write guarantee.
     *  Tickets can outlive bridge destruction, which drains accepted work.
     *  Dropping a ticket does not cancel work; caller must not mutate results.
     *  Like all bridge methods, submission must not race bridge destruction. */
    ProbeBatchTicket try_probe_batch(std::span<const DiskKVIdentity> ids, DiskKVKind kind);
    static bool poll(const ProbeBatchTicket& ticket) noexcept {
        return ticket->ready.load(std::memory_order_acquire);
    }
    SpillTicket try_submit(const DiskKVIdentity& id, DiskKVKind kind,
                           std::span<const std::byte> bytes);
    // Busy preserves bytes; accepted jobs own their payload through completion.
    SpillTicket try_submit_owned(const DiskKVIdentity& id, DiskKVKind kind,
                                 std::vector<std::byte>& bytes);
    // Maintenance lane: same ownership contract, but workers drain it first.
    SpillTicket try_submit_owned_priority(const DiskKVIdentity& id, DiskKVKind kind,
                                          std::vector<std::byte>& bytes);
    static SpillStatus poll(const SpillTicket& ticket) noexcept {
        return ticket->status.load(std::memory_order_acquire);
    }

    /** Wait until queued jobs and their threshold/idle index flushes finish.
     *  Test/diagnostics; tickets alone only acknowledge readable data. */
    // CPU-only worker borrows dst until acquire-poll is terminal. Cancellation
    // does NOT revoke the borrow. Null = busy; no store lock on submission.
    ReadTicket try_read(const DiskKVIdentity& id, DiskKVKind kind, std::span<std::byte> dst);
    // Maintenance lane for the read-back that validates a prepared closure.
    ReadTicket try_read_priority(const DiskKVIdentity& id, DiskKVKind kind, std::span<std::byte> dst);
    static DiskReadResult poll(const ReadTicket& ticket) noexcept {
        return ticket->result.load(std::memory_order_acquire);
    }
    void wait_idle();

    /** Read one previously spilt page into `dst` (family stride bytes).
     *  Synchronous (reads on the calling thread; the async queue is spill-only),
     *  so the read-side restore never waits on the background writer. Returns
     *  false on miss/evict/corrupt -> caller recomputes. */
    [[nodiscard]] bool restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                    std::span<std::byte> dst) const;

    /** Permanently drop a page that is indexed but unreadable (stale row,
     *  failed CRC). Self-heal for the restore path: the probe's chain walk
     *  rejects the frontier from then on, so one bad page can never wedge
     *  more than a single request. */
    bool drop_page(const DiskKVIdentity& id, DiskKVKind kind);

    /** True if this identity's page is restorable on disk right now. Cheap
     *  (header check). */
    [[nodiscard]] bool contains(const DiskKVIdentity& id, DiskKVKind kind) const;

    /** Read-only benefit probe: longest contiguous prefix (from page 0) of
     *  `page_ids` that is restorable now. Admission-time Root-path requests
     *  use this to measure how much recompute the tier COULD save. */
    [[nodiscard]] std::size_t probe_prefix(const std::vector<DiskKVIdentity>& page_ids,
                                           DiskKVKind kind) const;

    /** Append one probe record as a JSONL line next to the disk files. */
    void record_probe(std::uint32_t prompt_tokens, std::uint32_t restorable_tokens) const;

    /** Refresh LRU position after a successful restore. */
    void touch(const DiskKVIdentity& id, DiskKVKind kind);

    /** Refresh ONLY the in-memory LRU position (no header write, no index
     *  dirtying): the engine's liveness sweep re-ranks eviction by session
     *  liveness through this, thousands of pages per pass. */
    void touch_lru(const DiskKVIdentity& id, DiskKVKind kind);

    /** Frontiers of every live StateImage page (the state image at frontier E
     *  is keyed digest(E) — the same key as the partial tail KV page at E, so
     *  a restorable boundary is exactly a live state frontier whose digest the
     *  incoming request's own chain also produces). Empty when the state family
     *  is disabled. */
    [[nodiscard]] std::vector<std::uint32_t> live_state_frontiers() const;

    [[nodiscard]] DiskKVBridgeStats stats() const;
    [[nodiscard]] std::size_t used_bytes(DiskKVKind kind) const;
    [[nodiscard]] std::uint32_t slot_count(DiskKVKind kind) const;
    [[nodiscard]] const std::string& path(DiskKVKind kind) const;

private:
    struct Family {
        std::unique_ptr<DiskKVStore> store;
        std::size_t stride = 0;
        std::string path;
    };

    void worker_loop();
    /** Apply one queued job to the store (worker thread / destructor drain). */
    void spill_queued(SpillJob& job);
    /** Deadline-bounded large read (state images): runs the store read into
     *  attempt-owned memory on a transient thread; on timeout the attempt is
     *  abandoned (IOFailure) so the caller never blocks on a wedged volume. */
    DiskReadResult bounded_large_read(SpillJob& job);
    /** Deadline-bounded synchronous write used by engine-thread spill paths
     *  (seam anchors, owner-spill state): the payload is copied first, so an
     *  abandoned attempt never touches engine memory. False on timeout. */
    bool bounded_sync_write(const DiskKVIdentity& id, DiskKVKind kind,
                            std::span<const std::byte> bytes);

    Family& family(DiskKVKind kind);
    const Family& family(DiskKVKind kind) const;
    [[nodiscard]] static std::uint32_t kind_index(DiskKVKind kind) {
        return static_cast<std::uint32_t>(kind);
    }

    Options opts_;
    bool    enabled_ = false;
    Family families_[3];
    mutable std::mutex mu_;   // guards stats (stores are self-locked)
    mutable DiskKVBridgeStats stats_;
    // Index-durability batching: the DATA slots are self-describing (magic +
    // identity + CRC) and the store rebuilds a missing index by scanning them,
    // so an atomic index rewrite per page is pure overhead (a crash loses at
    // most the last batch — the scan recovers the durable slots).
    // Guarded by qmu_: all three workers increment/claim batches. Reset BEFORE
    // flushing outside qmu_; in_flight_ covers that I/O through completion.
    std::uint32_t flush_pending_ = 0;
    // Bounded async spill queue + writer thread.
    std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<SpillJob> queue_;
    // Maintenance lane drained before the general queue; same capacity budget.
    std::deque<SpillJob> priority_queue_;
    std::vector<std::thread> workers_;
    bool quit_ = false;
    std::size_t in_flight_ = 0;  // jobs pulled from the queue, not yet applied
    static constexpr std::size_t kQueueCap = 512;  // ~780 MiB in flight at 1.5 MiB/page
    // Parallel writers: the store's unlocked data section lets N writers copy
    // pages concurrently; the SSD needs several in-flight streams to reach its
    // bandwidth (single-writer was ~180 MB/s of a ~2 GB/s device through the
    // bind-mount layer). Bookkeeping stays serialized by the store lock.
    // Three workers could not keep the NVMe busy even when every reader held the
    // store lock only briefly: four concurrent streams already reach ~880MB/s on
    // this volume against 421MB/s single-stream. Reads and writes share these.
    static constexpr std::size_t kWorkerCount = 8;
    // Bounded large IO. The mounted volume has wedged single large preads and
    // pwrites indefinitely (twice in production, freezing the engine's single
    // context transaction past recovery). Large reads (state images) run on a
    // transient thread with a deadline; an abandoned attempt settles as
    // IOFailure so the transaction degrades to recompute instead of wedging.
    // Small pages stay on the direct path (thread spawn per 1.5MB page would
    // dominate); the per-syscall chunk cap in the store keeps each wait short.
    static constexpr std::size_t kLargeIOBytes = 8U << 20;
    static constexpr std::uint32_t kMaxAbandonedIO = 16;
    std::atomic<std::uint32_t> abandoned_ios_{0};
};

} // namespace ninfer
