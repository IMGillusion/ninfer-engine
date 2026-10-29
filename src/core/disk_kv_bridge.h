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
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
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
    // Set once at ticket creation. A Pending ticket older than the queue deadline means
    // no worker ever dequeued it (all workers wedged in deadline-less store IO): the
    // engine treats that as IOFailure and recomputes honestly instead of waiting forever
    // — the deadline on the dequeued side (bounded_large_read) never starts for a ticket
    // that is never dequeued. (observed: phase-0 seed read Pending for 300s, exit 75.)
    std::chrono::steady_clock::time_point submitted_at{std::chrono::steady_clock::now()};
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

    /** Fire-and-forget large spill (state images ~146 MiB): copies `bytes`
     *  into a ring-owned buffer on the calling thread, then hands the write
     *  to a transient thread and returns immediately. The persisted page
     *  becomes restorable atomically when the write lands; until then
     *  probe/contains simply miss and restores degrade to an older frontier
     *  plus delta recompute. `wait_for_slot`: the load-bearing owner-death
     *  states fall back to the bounded synchronous write when every ring
     *  slot is busy - a silently skipped endpoint state orphans the whole
     *  persisted chain (observed: agent hit rate 82% -> 51% with a 4-slot
     *  ring under six concurrent chains). Best-effort anchors pass false.
     *  False (skip) only when !wait_for_slot, the family is disabled, the
     *  volume breaker is tripped, or this identity is already restorable.
     *  `bytes` only needs to live until this call returns. */
    bool spill_page_async(const DiskKVIdentity& id, DiskKVKind kind,
                          std::span<const std::byte> bytes, bool wait_for_slot);

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
    // Volume circuit breaker: tripped when a queue-side deadline expires (workers
    // wedged on a hung volume). While tripped, all NEW submissions are rejected
    // instantly (null / false) so callers degrade to recompute instead of piling
    // more IO behind the wedge; after the window the breaker auto-closes (lazy,
    // on the next check) and submissions resume.
    void trip_volume_breaker(std::chrono::milliseconds window) noexcept;
    [[nodiscard]] bool volume_breaker_tripped() const noexcept;
    static DiskReadResult poll(const ReadTicket& ticket) noexcept {
        return ticket->result.load(std::memory_order_acquire);
    }
    // Queue-side deadline: a ticket still Pending after `queue_deadline` since submission
    // was never dequeued by any worker (every worker wedged in deadline-less store IO on a
    // hung volume). The dequeued-side deadline (bounded_large_read, 60s) cannot cover this
    // — it only starts once a worker picks the job up. Treat the expiry as IOFailure so the
    // engine degrades to an honest full recompute instead of waiting forever.
    // (observed: phase-0 seed read Pending for 300s, sentinel exit 75.)
    static DiskReadResult poll(const ReadTicket& ticket,
                               std::chrono::steady_clock::time_point now,
                               std::chrono::milliseconds queue_deadline) noexcept {
        const DiskReadResult result = ticket->result.load(std::memory_order_acquire);
        if (result == DiskReadResult::Pending &&
            now - ticket->submitted_at > queue_deadline) {
            return DiskReadResult::IOFailure;
        }
        return result;
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

    /** True if this identity's page is restorable on disk right now - or is
     *  a SUBMITTED write still in flight (promise). The admission probe walks
     *  chains through this, so a chain whose checkpoint was handed off
     *  moments ago is selectable; the seed's read of a promised page waits
     *  for the write to land instead of mis-failing the whole restore
     *  (observed: back-to-back chain rounds always probed before their own
     *  checkpoint landed - every round recomputed from zero). */
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

    void worker_loop(bool read_worker);
    /** Apply one queued job to the store (worker thread / destructor drain). */
    void spill_queued(SpillJob& job);
    /** Deadline-bounded large read (state images): runs the store read into
     *  attempt-owned memory on a transient thread; on timeout the attempt is
     *  abandoned (IOFailure) so the caller never blocks on a wedged volume. */
    DiskReadResult bounded_large_read(SpillJob& job);
    /** Bounded synchronous write used as spill_page_async's saturation
     *  fallback: the payload is copied first, so an abandoned attempt never
     *  touches engine memory. False on timeout. */
    bool bounded_sync_write(const DiskKVIdentity& id, DiskKVKind kind,
                            std::span<const std::byte> bytes);
    /** Ring slot for spill_page_async: one large payload buffer per in-flight
     *  fire-and-forget write. Copy happens on the calling thread, so the
     *  source scratch can be released (or recycled by the next request) the
     *  moment this call returns. */
    struct AsyncWriteSlot {
        std::unique_ptr<std::byte[]> bytes;
        std::size_t capacity = 0;
        bool busy = false;
    };
    static constexpr std::size_t kAsyncWriteSlots = 12;
    std::mutex async_mu_;
    std::array<AsyncWriteSlot, kAsyncWriteSlots> async_slots_{};
    // Live fire-and-forget writers; the destructor gives them a bounded
    // window (write deadline) before the stores are torn down.
    std::atomic<std::size_t> async_live_{0};
    // Submitted-but-not-landed write identities (the promise set). Counted:
    // the same identity can be submitted twice (turn-end checkpoint plus a
    // dying-owner re-spill). contains() reports promised identities as
    // present so a probe never rejects a chain whose head write is in
    // flight; the seed's read of a promised page waits for it to land.
    // Lock ordering: qmu_ -> promised_mu_ (never the reverse).
    struct IdentityKey {
        std::uint64_t lo;
        std::uint64_t hi;
        std::uint32_t tag;
        std::uint32_t frontier;
        bool operator<(const IdentityKey& o) const {
            return std::tuple(lo, hi, tag, frontier) <
                   std::tuple(o.lo, o.hi, o.tag, o.frontier);
        }
    };
    static IdentityKey identity_key(const DiskKVIdentity& id) noexcept {
        return {id.lo, id.hi, id.tag, id.frontier};
    }
    mutable std::mutex promised_mu_;
    std::map<IdentityKey, int> promised_;
    void promise_add(const DiskKVIdentity& id) {
        std::lock_guard<std::mutex> lock(promised_mu_);
        promised_[identity_key(id)] += 1;
    }
    void promise_done(const DiskKVIdentity& id) noexcept {
        std::lock_guard<std::mutex> lock(promised_mu_);
        const auto it = promised_.find(identity_key(id));
        if (it != promised_.end() && --(it->second) <= 0) { promised_.erase(it); }
    }
    [[nodiscard]] bool promised(const DiskKVIdentity& id) const {
        std::lock_guard<std::mutex> lock(promised_mu_);
        return promised_.count(identity_key(id)) != 0;
    }
    /** Read-side wait for a promised write to land (bounded by the write
     *  deadline). Only reads racing their own chain's checkpoint pay it. */
    void wait_promise_clear(const DiskKVIdentity& id);

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
    // Request-path lane (seed restore reads) drained before maintenance: a
    // checkpoint/eviction flood used to park in front of the priority lane and
    // stretch every admission's seed to 2.5-5s under load - the single
    // materialization transaction gates admission, so that latency multiplies
    // across every queued request. Same capacity budget.
    std::deque<SpillJob> seed_queue_;
    // Sum of all three lanes; only meaningful under qmu_.
    [[nodiscard]] std::size_t queued_jobs() const noexcept {
        return queue_.size() + priority_queue_.size() + seed_queue_.size();
    }
    // Read-side outstanding (queued + executing read jobs); only meaningful
    // under qmu_. Read admission is judged against kReadBudget ALONE, never
    // against the total: a QLC GC stall parking every write worker must not
    // push the shared total to the cap and refuse the seed's page reads -
    // that refusal stalls the single materialization transaction, and the
    // stall gates admission for every queued request.
    std::size_t queued_reads_ = 0;
    [[nodiscard]] std::size_t read_outstanding() const noexcept {
        return queued_reads_ + in_flight_reads_;
    }
    // Seed-restore read budget: two 128-page seed chunks plus headroom for
    // state-image reads and probe read-backs. Writes cannot eat into it.
    static constexpr std::size_t kReadBudget = 320;
    std::vector<std::thread> workers_;
    bool quit_ = false;
    std::size_t in_flight_ = 0;  // jobs pulled from the queue, not yet applied
    static constexpr std::size_t kQueueCap = 512;  // ~780 MiB in flight at 1.5 MiB/page
    // Parallel writers: the store's unlocked data section lets N writers copy
    // pages concurrently; the SSD needs several in-flight streams to reach its
    // bandwidth (single-writer was ~180 MB/s of a ~2 GB/s device through the
    // bind-mount layer). Bookkeeping stays serialized by the store lock.
    // Twelve workers: 8 read (the Ti600's measured read optimum is QD8 -
    // 2156 MB/s; QD5 read workers left a third of the bandwidth on the
    // table) + 4 write for the phase-gated checkpoint stream.
    static constexpr std::size_t kWorkerCount = 12;
    // Read/write phase gate: while seed reads occupy the device, write
    // workers (and the direct writers) hold so the QLC device runs reads
    // and writes in phases. Measured on the Ti600: pure reads 2156 MB/s;
    // with a state-image write in flight the same reads collapse to
    // 75-150 MB/s and the write itself to ~3 MB/s. Promise-waiting reads
    // are excluded from the occupancy, so the write that clears their
    // promise is never gated behind them (no deadlock, no timeout needed).
    /** Hold the caller while non-promise seed reads are in flight. */
    void wait_read_phase_clear();
    // Read/write worker partition: request-path reads (seed page + state
    // reads) execute on dedicated read workers and can never be stuck
    // behind an in-flight checkpoint/spill write that a QLC GC stall is
    // stretching to tens of seconds. Queue priority alone cannot give
    // that isolation - a queued read still waits for every busy write
    // worker to finish its current job. Tunable via NINFER_L3_READ_WORKERS.
    static std::size_t read_worker_count() noexcept {
        static const std::size_t value = [] {
            const char* s = std::getenv("NINFER_L3_READ_WORKERS");
            std::size_t v = s != nullptr ? std::strtoul(s, nullptr, 10) : 8;
            return v < kWorkerCount ? v : kWorkerCount - 1;
        }();
        return value;
    }
    std::size_t in_flight_reads_ = 0;  // read jobs popped, not yet applied
    // Reads currently inside wait_promise_clear: they hold a worker but are
    // WAITING for a write, so they do not occupy the device. The write-side
    // phase gate subtracts them from in_flight_reads_, which both removes
    // the gate's timeout (true read/write phasing) and breaks the
    // promise/gate deadlock without the 8s escape hatch.
    std::size_t promise_wait_reads_ = 0;
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
    // Volume circuit breaker: steady-clock ticks (time_since_epoch count) of the
    // trip deadline, 0 = closed. Single atomic, lock-free check on every submission.
    std::atomic<std::int64_t> breaker_until_ticks_{0};
};

} // namespace ninfer
