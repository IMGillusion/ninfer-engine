#include "core/disk_kv_bridge.h"

#include "core/disk_kv_log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ninfer {

namespace {

std::string family_path(const std::string& base, DiskKVKind kind) {
    const char* name = kind == DiskKVKind::MainKV ? "main"
                     : kind == DiskKVKind::BackendKV ? "backend" : "state";
    return base + "/diskkv_" + name;
}

// Split the total budget across enabled families: 65% main / 10% backend /
// 25% state. Backend (MTP draft) pages are 16x smaller per token than main
// (one layer, 4 heads), so 10% holds several times the draft working set;
// state images are the restore bottleneck — the probe requires the state
// image at the frontier, so an evicted state image orphans a complete KV
// chain (measured: 43 slots vs 24+ sessions × endpoint+rewrite+anchors).
std::size_t family_budget(std::size_t total, std::size_t stride, std::uint32_t share,
                         std::uint32_t denominator) {
    if (stride == 0) { return 0; }
    const std::size_t bytes = total * share / denominator;
    return (bytes / stride) * stride;
}

} // namespace

DiskKVBridge::DiskKVBridge(Options opts) : opts_(std::move(opts)) {
    if (opts_.base_path.empty() || opts_.capacity_bytes == 0) {
        enabled_ = false;  // explicit off
        return;
    }
    if (opts_.main_page_stride == 0) {
        throw std::invalid_argument("disk kv bridge needs main_page_stride");
    }
    auto make_family = [&](DiskKVKind kind, std::size_t family_stride, std::size_t budget) {
        Family& f       = families_[kind_index(kind)];
        f.stride        = family_stride;
        f.path          = family_path(opts_.base_path, kind);
        if (family_stride == 0 || budget == 0) { return; }  // family disabled
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(f.path).parent_path(), ec);
        DiskKVStore::Options sopts;
        sopts.path           = f.path;
        sopts.slot_size      = family_stride;
        sopts.capacity_bytes = budget;
        sopts.max_slots      = 0;  // derived from capacity_bytes
        sopts.verify_crc     = opts_.verify_crc;
        sopts.defer_index_updates = true;
        f.store = std::make_unique<DiskKVStore>(sopts);
    };
    const std::size_t total = opts_.capacity_bytes;
    const std::uint32_t main_share    = opts_.backend_page_stride ? 65U : 75U;
    const std::uint32_t backend_share = opts_.backend_page_stride ? 10U : 0U;
    const std::uint32_t main_denom    = 100U;
    make_family(DiskKVKind::MainKV, opts_.main_page_stride,
                family_budget(total, opts_.main_page_stride, main_share, main_denom));
    make_family(DiskKVKind::BackendKV, opts_.backend_page_stride,
                family_budget(total, opts_.backend_page_stride, backend_share, 100U));
    // State images are large single blobs (~146 MiB each for the 48-layer GDN
    // state). Give them a slice big enough to hold several, so a restarted
    // multi-turn session can restore its recurrent state at a shared frontier.
    make_family(DiskKVKind::StateImage, opts_.state_page_stride,
                family_budget(total, opts_.state_page_stride, 100U - main_share - backend_share, 100U));
    enabled_ = families_[0].store != nullptr;
    if (!enabled_) {
        throw std::invalid_argument("disk kv bridge budget too small for one main page");
    }
    workers_.reserve(kWorkerCount);
    for (std::size_t i = 0; i < kWorkerCount; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

DiskKVBridge::~DiskKVBridge() {
    {
        std::lock_guard<std::mutex> lock(qmu_);
        quit_ = true;
    }
    qcv_.notify_all();
    for (std::thread& w : workers_) {
        if (w.joinable()) { w.join(); }
    }
    // Drain whatever the worker already pulled off the queue so no page is
    // lost between join and destruction (best effort; store is still alive).
    while (true) {
        SpillJob job;
        {
            std::lock_guard<std::mutex> lock(qmu_);
            if (queue_.empty() && priority_queue_.empty()) { break; }
            job = std::move(priority_queue_.empty() ? queue_.front() : priority_queue_.front());
            if (priority_queue_.empty()) { queue_.pop_front(); } else { priority_queue_.pop_front(); }
        }
        spill_queued(job);
    }
    // Durability barrier before the stores go away (batched index updates).
    for (Family& fam : families_) {
        if (fam.store != nullptr) { fam.store->flush_index(); }
    }
}

void DiskKVBridge::wait_idle() {
    {
        std::unique_lock<std::mutex> lock(qmu_);
        qcv_.wait(lock, [this] { return queue_.empty() && priority_queue_.empty() && in_flight_ == 0; });
    }
    // Worker keeps in_flight nonzero through its idle flush.
}

namespace {
// NINFER_BRIDGE_TRACE=1 logs submit/dequeue/write milestones for stall forensics.
inline bool bridge_trace() {
    static const bool on = std::getenv("NINFER_BRIDGE_TRACE") != nullptr;
    return on;
}

// Deadlines for the bounded large-IO paths (0 keeps the bound at the default).
std::uint64_t env_ms(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) { return fallback; }
    return static_cast<std::uint64_t>(parsed);
}
std::chrono::milliseconds read_deadline() {
    return std::chrono::milliseconds(env_ms("NINFER_L3_READ_TIMEOUT_MS", 60000));
}
std::chrono::milliseconds write_deadline() {
    return std::chrono::milliseconds(env_ms("NINFER_L3_WRITE_TIMEOUT_MS", 30000));
}
} // namespace

void DiskKVBridge::worker_loop() {
    while (true) {
        SpillJob job;
        bool have = false;
        try {
        {
            std::unique_lock<std::mutex> lock(qmu_);
            qcv_.wait(lock, [this] { return quit_ || !queue_.empty() || !priority_queue_.empty(); });
            if (quit_ && queue_.empty() && priority_queue_.empty()) { return; }
            if (queue_.empty() && priority_queue_.empty()) { continue; }
            job = std::move(priority_queue_.empty() ? queue_.front() : priority_queue_.front());
            if (priority_queue_.empty()) { queue_.pop_front(); } else { priority_queue_.pop_front(); }
            in_flight_ += 1;
            have = true;
            if (bridge_trace()) {
                disk_kv_logf('D', "l3-bridge",
                             "dequeue | id=%llu/%llu kind=%u queued=%zu in_flight=%zu",
                             static_cast<unsigned long long>(job.id.lo), static_cast<unsigned long long>(job.id.hi),
                             static_cast<unsigned>(job.kind), queue_.size(), in_flight_);
            }
        }
        if (!have) {
            std::lock_guard<std::mutex> lock(qmu_);
            if (quit_) { return; }
            continue;
        }
        spill_queued(job);
        bool flush = false;
        {
            std::unique_lock<std::mutex> lock(qmu_);
            // Consecutive resumable tickets have scheduler gaps. Coalesce them;
            // waiting releases qmu and never delays try_submit on the engine.
            if (queue_.empty() && priority_queue_.empty() && flush_pending_ != 0 && !quit_) {
                qcv_.wait_for(lock, std::chrono::milliseconds(10),
                              [this] { return quit_ || !queue_.empty() || !priority_queue_.empty(); });
            }
            flush = queue_.empty() && priority_queue_.empty() && flush_pending_ != 0;
            // Claim this batch under the same lock as every increment. Writes
            // completing after the claim belong to the next batch; never clear
            // their count after the (unlocked) I/O.
            if (flush) { flush_pending_ = 0; }
        }
        if (flush) {
            // The burst drained: land the batched index updates while idle.
            for (Family& fam : families_) {
                if (fam.store != nullptr) { fam.store->flush_index(); }
            }
        }
        {
            std::lock_guard<std::mutex> lock(qmu_);
            in_flight_ -= 1;
        }
        qcv_.notify_all();  // wait_idle includes flush; enqueue never holds a disk lock
        } catch (const std::exception& e) {
            // A worker-side I/O/allocation failure must never terminate the
            // engine — and must never orphan an accepted ticket. Settle every
            // completion this job owns as Failed; otherwise a failed spill
            // drains forever and the engine boundary livelocks (observed: the
            // STALL heartbeat with tickets=32 unacknowledged for 6+ minutes).
            if (have) {
                if (job.completion) { job.completion->status.store(SpillStatus::Failed, std::memory_order_release); }
                if (job.read) { job.read->result.store(DiskReadResult::IOFailure, std::memory_order_release); }
                if (job.probe_batch) { job.probe_batch->ready.store(true, std::memory_order_release); }
            }
            std::lock_guard<std::mutex> lock(qmu_);
            in_flight_ -= (in_flight_ > 0 ? 1 : 0);
            qcv_.notify_all();
        } catch (...) {
            if (have) {
                if (job.completion) { job.completion->status.store(SpillStatus::Failed, std::memory_order_release); }
                if (job.read) { job.read->result.store(DiskReadResult::IOFailure, std::memory_order_release); }
                if (job.probe_batch) { job.probe_batch->ready.store(true, std::memory_order_release); }
            }
            std::lock_guard<std::mutex> lock(qmu_);
            in_flight_ -= (in_flight_ > 0 ? 1 : 0);
            qcv_.notify_all();
        }
    }
}

bool DiskKVBridge::spill_page(const DiskKVIdentity& id, DiskKVKind kind,
                              std::span<const std::byte> bytes) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        // touch() returns presence AND refreshes LRU residency: a dedupe hit
        // proves the page is still wanted (its owner chain is being re-spilled),
        // so the verified page must stop aging toward eviction. (contains()
        // deliberately never refreshed — the reason live sessions' deep
        // prefixes used to go cold on disk while their sessions stayed hot.)
        if (f.store->touch(id)) {
            stats_.spill_dups += 1;
            return true;  // already restorable; shared-prefix dedupe
        }
    }
    std::lock_guard<std::mutex> lock(qmu_);
    // kQueueCap bounds TOTAL in-flight work: queued general, queued maintenance,
    // and jobs already dequeued into a worker.
    if (queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        std::lock_guard<std::mutex> stats_lock(mu_);
        stats_.queue_drops += 1;
        return false;  // backpressure: page won't be restorable (recompute later)
    }
    queue_.emplace_back(SpillJob{id, kind, {bytes.begin(), bytes.end()}, {}, {}, false, {}, {}});
    qcv_.notify_all();
    return true;
}

bool DiskKVBridge::spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes,
                                   std::chrono::milliseconds timeout) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (f.store->touch(id)) {  // presence + LRU refresh (see spill_page)
            stats_.spill_dups += 1;
            return true;  // already restorable; shared-prefix dedupe
        }
    }
    std::unique_lock<std::mutex> lock(qmu_);
    // Bounded backpressure: the owner-death sweep must land a CONTIGUOUS
    // chain, so wait for a free slot instead of dropping the page. The
    // worker notifies qcv_ after every pop; on shutdown we bail out.
    if (!qcv_.wait_for(lock, timeout, [this] {
            return quit_ || queue_.size() + priority_queue_.size() + in_flight_ < kQueueCap;
        })) {
        std::lock_guard<std::mutex> stats_lock(mu_);
        stats_.queue_drops += 1;
        return false;
    }
    if (quit_) { return false; }
    queue_.emplace_back(SpillJob{id, kind, {bytes.begin(), bytes.end()}, {}, {}, false, {}, {}});
    qcv_.notify_all();
    return true;
}

ReadTicket DiskKVBridge::try_read(const DiskKVIdentity& id, DiskKVKind kind,
                                  std::span<std::byte> dst) {
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ ||
        queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        return {};
    }
    auto ticket = std::make_shared<ReadCompletion>();
    if (!enabled_ || kind_index(kind) >= std::size(families_) || !family(kind).store) {
        ticket->result.store(DiskReadResult::Disabled, std::memory_order_release);
        return ticket;
    }
    if (dst.size() != family(kind).stride) {
        ticket->result.store(DiskReadResult::BadSize, std::memory_order_release);
        return ticket;
    }
    SpillJob job{};
    job.id = id; job.kind = kind; job.read = ticket; job.read_dst = dst;
    queue_.push_back(std::move(job));
    lock.unlock();
    qcv_.notify_one();
    return ticket;
}

void DiskKVBridge::spill_queued(SpillJob& job) {
    if (job.read) {
        DiskReadResult result = DiskReadResult::IOFailure;
        try {
            if (job.read_dst.size() >= kLargeIOBytes) {
                result = bounded_large_read(job);
            } else {
                result = family(job.kind).store->read_page_result(job.id, job.read_dst);
            }
            std::lock_guard<std::mutex> lock(mu_);
            if (result == DiskReadResult::Success) {
                ++stats_.restores; stats_.restore_bytes += job.read_dst.size();
            } else { ++stats_.restore_misses; }
        } catch (...) {}
        // Final access to borrowed bytes precedes release acknowledgement.
        job.read_dst = {};
        job.read->result.store(result, std::memory_order_release);
        return;
    }
    if (job.probe_batch) {
        // No queue/stats lock spans store access. Failures are per-page;
        // continue probing the remaining identities and always publish ready.
        auto& batch = *job.probe_batch;
        std::uint64_t present = 0;
        for (std::size_t i = 0; i < job.probe_ids.size(); ++i) {
            try {
                // Verification refreshes residency (see spill_page): Present
                // proves the page is still referenced by a live owner chain,
                // so per-turn checkpoint probes keep active chains hot.
                const bool hit = family(job.kind).store->touch(job.probe_ids[i]);
                batch.results[i] = hit ? SpillStatus::Present : SpillStatus::Missing;
                present += hit;
            } catch (...) {
                batch.results[i] = SpillStatus::Failed;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.spill_dups += present;
        }
        batch.ready.store(true, std::memory_order_release);
        return;
    }
    Family& f = family(job.kind);
    const auto finish = [&](SpillStatus status) {
        // Release owned capacity BEFORE ack permits the producer's next buffer.
        // Borrowed owner-spill jobs have an empty vector and are unchanged.
        std::vector<std::byte>().swap(job.bytes);
        if (job.completion) { job.completion->status.store(status, std::memory_order_release); }
    };
    try {
    if (job.probe) {
        // Verification refreshes residency (see spill_page): Present proves
        // the page is still referenced by a live owner chain.
        const bool present = f.store && f.store->touch(job.id);
        if (present) { std::lock_guard<std::mutex> lock(mu_); ++stats_.spill_dups; }
        finish(present ? SpillStatus::Present : SpillStatus::Missing);
        return;
    }
    const auto bytes = job.borrowed.empty() ? std::span<const std::byte>(job.bytes) : job.borrowed;
    if (f.store == nullptr || bytes.size() != f.stride) {
        if (f.store != nullptr) {
            disk_kv_logf('E', "l3-bridge",
                         "FAIL kind=%u | stride mismatch bytes=%zu family=%zu | id=%llu/%llu/%llu",
                         static_cast<unsigned>(job.kind), bytes.size(), f.stride,
                         static_cast<unsigned long long>(job.id.lo),
                         static_cast<unsigned long long>(job.id.hi),
                         static_cast<unsigned long long>(job.id.tag));
        }
        finish(SpillStatus::Failed);
        return;
    }
    std::vector<std::uint32_t> evicted;
    if (!f.store->upsert_page(job.id, bytes, &evicted)) {
        // The disk tier is a pure cache: a failed write (store full with no
        // evictable victim, or I/O error) must degrade to "page not cached"
        // — a future restore recompute — never a request failure. Settle as
        // Dropped so the demotion completes and the host extents release.
        disk_kv_logf('W', "l3-bridge",
                     "DROP upsert=false | kind=%u size=%zu | id=%llu/%llu/%llu",
                     static_cast<unsigned>(job.kind), bytes.size(),
                     static_cast<unsigned long long>(job.id.lo),
                     static_cast<unsigned long long>(job.id.hi),
                     static_cast<unsigned long long>(job.id.tag));
        finish(SpillStatus::Dropped);
        return;
    }
    // Batched index durability (see flush_pending_ in the header): a per-page
    // atomic index rewrite dominated large owner-death sweeps.
    bool flush = false;
    {
        std::lock_guard<std::mutex> lock(qmu_);
        flush = ++flush_pending_ >= 64;
        if (flush) { flush_pending_ = 0; }
    }
    if (flush) {
        for (Family& fam : families_) {
            if (fam.store != nullptr) { fam.store->flush_index(); }
        }
    }
    std::lock_guard<std::mutex> lock(mu_);
    stats_.spills += 1;
    stats_.spill_bytes += bytes.size();
    stats_.evicted_slots += evicted.size();
    finish(SpillStatus::Written);
    } catch (const std::exception& e) {
        disk_kv_logf('E', "l3-bridge",
                     "FAIL exception | kind=%u id=%llu/%llu/%llu | what=%s",
                     static_cast<unsigned>(job.kind),
                     static_cast<unsigned long long>(job.id.lo),
                     static_cast<unsigned long long>(job.id.hi),
                     static_cast<unsigned long long>(job.id.tag), e.what());
        finish(SpillStatus::Failed);
    } catch (...) {
        disk_kv_logf('E', "l3-bridge", "FAIL unknown exception | kind=%u",
                     static_cast<unsigned>(job.kind));
        finish(SpillStatus::Failed);
    }
}

ProbeBatchTicket DiskKVBridge::try_probe_batch(std::span<const DiskKVIdentity> ids,
                                               DiskKVKind kind) {
    if (ids.empty() || ids.size() > kDiskKVProbeBatchMax) {
        throw std::invalid_argument("disk kv probe batch must contain 1..32 identities");
    }
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ ||
        queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        return {};
    }
    auto ticket = std::make_shared<ProbeBatchCompletion>();
    ticket->size = ids.size();
    ticket->results.fill(SpillStatus::Failed);
    if (!enabled_ || kind_index(kind) >= std::size(families_) || !family(kind).store) {
        ticket->ready.store(true, std::memory_order_release);
        return ticket;
    }
    SpillJob job{};
    job.kind = kind;
    job.probe_ids.assign(ids.begin(), ids.end());
    job.probe_batch = ticket;
    queue_.push_back(std::move(job));
    lock.unlock();
    qcv_.notify_one();
    return ticket;
}

SpillTicket DiskKVBridge::try_probe(const DiskKVIdentity& id, DiskKVKind kind) {
    if (kind_index(kind) >= std::size(families_)) { return {}; }
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ ||
        queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        return {};
    }
    auto ticket = std::make_shared<SpillCompletion>();
    SpillJob job{}; job.id = id; job.kind = kind; job.completion = ticket; job.probe = true;
    queue_.push_back(std::move(job));
    qcv_.notify_one();
    return ticket;
}

SpillTicket DiskKVBridge::try_submit(const DiskKVIdentity& id, DiskKVKind kind,
                                    std::span<const std::byte> bytes) {
    if (kind_index(kind) >= std::size(families_)) { return {}; }
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ ||
        queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        return {};
    }
    auto ticket = std::make_shared<SpillCompletion>();
    SpillJob job{}; job.id = id; job.kind = kind; job.completion = ticket; job.borrowed = bytes;
    queue_.push_back(std::move(job));
    if (bridge_trace()) {
        disk_kv_logf('D', "l3-bridge", "submit | id=%llu/%llu kind=%u queued=%zu in_flight=%zu",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                     static_cast<unsigned>(kind), queue_.size(), in_flight_);
    }
    qcv_.notify_one();
    return ticket;
}

SpillTicket DiskKVBridge::try_submit_owned(const DiskKVIdentity& id, DiskKVKind kind,
                                         std::vector<std::byte>& bytes) {
    if (kind_index(kind) >= std::size(families_)) { return {}; }
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ ||
        queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) {
        return {};
    }
    auto ticket = std::make_shared<SpillCompletion>();
    // Allocate queue storage BEFORE transferring ownership (exception-safe retry).
    queue_.emplace_back();
    auto& job = queue_.back();
    job.id = id; job.kind = kind; job.completion = ticket;
    job.bytes.swap(bytes);
    qcv_.notify_one();
    return ticket;
}

SpillTicket DiskKVBridge::try_submit_owned_priority(const DiskKVIdentity& id, DiskKVKind kind,
                                                   std::vector<std::byte>& bytes) {
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ || queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) return {};
    auto ticket = std::make_shared<SpillCompletion>();
    // Allocate queue storage BEFORE transferring ownership (exception-safe retry).
    priority_queue_.emplace_back();
    auto& job = priority_queue_.back();
    job.id = id; job.kind = kind; job.completion = ticket; job.priority = true;
    job.bytes.swap(bytes);
    qcv_.notify_one();
    return ticket;
}

ReadTicket DiskKVBridge::try_read_priority(const DiskKVIdentity& id, DiskKVKind kind,
                                           std::span<std::byte> dst) {
    std::unique_lock<std::mutex> lock(qmu_, std::try_to_lock);
    if (!lock || quit_ || queue_.size() + priority_queue_.size() + in_flight_ >= kQueueCap) { return {}; }
    auto ticket = std::make_shared<ReadCompletion>();
    if (!enabled_ || kind_index(kind) >= std::size(families_) || !family(kind).store) {
        ticket->result.store(DiskReadResult::Disabled, std::memory_order_release);
        return ticket;
    }
    if (dst.size() != family(kind).stride) {
        ticket->result.store(DiskReadResult::BadSize, std::memory_order_release);
        return ticket;
    }
    SpillJob job{};
    job.id = id; job.kind = kind; job.read = ticket; job.read_dst = dst; job.priority = true;
    priority_queue_.push_back(std::move(job));
    lock.unlock();
    qcv_.notify_one();
    return ticket;
}

DiskReadResult DiskKVBridge::bounded_large_read(SpillJob& job) {
    // Shared attempt state: an abandoned thread may finish long after this
    // call returned, so everything it touches is heap-owned by the attempt.
    struct Attempt {
        std::unique_ptr<std::byte[]> buffer;
        std::atomic<bool> done{false};
        std::atomic<bool> abandoned{false};
        DiskReadResult result = DiskReadResult::IOFailure;
    };
    auto attempt     = std::make_shared<Attempt>();
    const auto n     = job.read_dst.size();
    attempt->buffer  = std::make_unique<std::byte[]>(n);
    const auto id    = job.id;
    const auto kind  = job.kind;
    auto* store      = family(kind).store.get();
    std::thread worker([this, attempt, id, kind, store, n] {
        attempt->result = store->read_page_result(id, {attempt->buffer.get(), n});
        attempt->done.store(true, std::memory_order_release);
        if (attempt->abandoned.exchange(true)) { abandoned_ios_.fetch_sub(1); }
    });
    const auto deadline = std::chrono::steady_clock::now() + read_deadline();
    bool finished = attempt->done.load(std::memory_order_acquire);
    while (!finished && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        finished = attempt->done.load(std::memory_order_acquire);
    }
    if (finished) {
        worker.join();
        std::memcpy(job.read_dst.data(), attempt->buffer.get(), n);
        return attempt->result;
    }
    // Timeout: settle as IOFailure and leave the syscall behind on its own
    // thread (it cannot be cancelled). The engine's scratch was never touched
    // — the read landed in the attempt buffer — so the transaction's fail()
    // path (honest full recompute) is safe while the abandoned attempt drains.
    if (!attempt->abandoned.exchange(true)) { abandoned_ios_.fetch_add(1); }
    disk_kv_logf('W', "l3-bridge",
                 "read abandoned | kind=%u bytes=%zu id=%llu/%llu outstanding=%u | volume wedged?",
                 static_cast<unsigned>(kind), n,
                 static_cast<unsigned long long>(id.lo),
                 static_cast<unsigned long long>(id.hi),
                 abandoned_ios_.load(std::memory_order_relaxed));
    worker.detach();
    return DiskReadResult::IOFailure;
}

bool DiskKVBridge::bounded_sync_write(const DiskKVIdentity& id, DiskKVKind kind,
                                      std::span<const std::byte> bytes) {
    struct Attempt {
        std::vector<std::byte> bytes;
        std::vector<std::uint32_t> evicted;
        std::atomic<bool> done{false};
        std::atomic<bool> abandoned{false};
        bool ok = false;
    };
    auto attempt    = std::make_shared<Attempt>();
    attempt->bytes.assign(bytes.begin(), bytes.end());
    auto* store     = family(kind).store.get();
    std::thread worker([this, attempt, id, store] {
        attempt->ok = store->upsert_page(id, attempt->bytes, &attempt->evicted);
        attempt->done.store(true, std::memory_order_release);
        if (attempt->abandoned.exchange(true)) { abandoned_ios_.fetch_sub(1); }
    });
    const auto deadline = std::chrono::steady_clock::now() + write_deadline();
    bool finished = attempt->done.load(std::memory_order_acquire);
    while (!finished && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        finished = attempt->done.load(std::memory_order_acquire);
    }
    if (!finished) {
        // Attempt owns its payload copy; the abandoned thread touches nothing
        // the caller owns. The anchor degrades silently (best-effort by design).
        if (!attempt->abandoned.exchange(true)) { abandoned_ios_.fetch_add(1); }
        disk_kv_logf('W', "l3-bridge",
                     "sync write abandoned | kind=%u bytes=%zu id=%llu/%llu outstanding=%u",
                     static_cast<unsigned>(kind), bytes.size(),
                     static_cast<unsigned long long>(id.lo),
                     static_cast<unsigned long long>(id.hi),
                     abandoned_ios_.load(std::memory_order_relaxed));
        worker.detach();
        return false;
    }
    worker.join();
    if (!attempt->ok) { return false; }
    const std::size_t evicted_n = attempt->evicted.size();
    family(kind).store->flush_index();
    std::lock_guard<std::mutex> lock(mu_);
    stats_.spills += 1;
    stats_.spill_bytes += bytes.size();
    stats_.evicted_slots += evicted_n;
    return true;
}

bool DiskKVBridge::spill_page_sync(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || bytes.size() != f.stride) { return false; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (f.store->contains(id)) {
            stats_.spill_dups += 1;
            return true;  // already restorable (dedupe)
        }
    }
    if (bytes.size() >= kLargeIOBytes) { return bounded_sync_write(id, kind, bytes); }
    std::vector<std::uint32_t> evicted;
    if (!f.store->upsert_page(id, bytes, &evicted)) { return false; }
    f.store->flush_index();
    std::lock_guard<std::mutex> lock(mu_);
    stats_.spills += 1;
    stats_.spill_bytes += bytes.size();
    stats_.evicted_slots += evicted.size();
    return true;
}

bool DiskKVBridge::drop_page(const DiskKVIdentity& id, DiskKVKind kind) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    Family& f = family(kind);
    if (f.store == nullptr) { return false; }
    return f.store->evict(id);
}

bool DiskKVBridge::restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                std::span<std::byte> dst) const {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr || dst.size() != f.stride) { return false; }
    if (!f.store->read_page(id, dst)) {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.restore_misses += 1;
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    stats_.restores += 1;
    stats_.restore_bytes += f.stride;
    return true;
}

bool DiskKVBridge::contains(const DiskKVIdentity& id, DiskKVKind kind) const {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return false; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return false; }
    return f.store->contains(id);
}

std::size_t DiskKVBridge::probe_prefix(const std::vector<DiskKVIdentity>& page_ids,
                                       DiskKVKind kind) const {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return 0; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return 0; }
    // Walk from page 0; the first miss stops the prefix.
    for (std::size_t p = 0; p < page_ids.size(); ++p) {
        if (!f.store->contains(page_ids[p])) { return p; }
    }
    return page_ids.size();
}

void DiskKVBridge::record_probe(std::uint32_t prompt_tokens,
                                std::uint32_t restorable_tokens) const {
    if (!enabled_ || restorable_tokens == 0 || opts_.base_path.empty()) { return; }
    try {
        const std::string path = opts_.base_path + "/diskkv_probe.jsonl";
        std::ofstream out(path, std::ios::app);
        if (!out) { return; }
        out << "{\"event\":\"diskkv_probe\",\"prompt_tokens\":" << prompt_tokens
            << ",\"restorable_tokens\":" << restorable_tokens << "}\n";
    } catch (...) {
    }
}

void DiskKVBridge::touch(const DiskKVIdentity& id, DiskKVKind kind) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return; }
    f.store->touch(id);
}

void DiskKVBridge::touch_lru(const DiskKVIdentity& id, DiskKVKind kind) {
    if (!enabled_ || kind_index(kind) >= std::size(families_)) { return; }
    const Family& f = family(kind);
    if (f.store == nullptr) { return; }
    f.store->touch_lru(id);
}

std::vector<std::uint32_t> DiskKVBridge::live_state_frontiers() const {
    std::vector<std::uint32_t> out;
    if (!enabled_) { return out; }
    const Family& f = family(DiskKVKind::StateImage);
    if (f.store == nullptr) { return out; }
    for (const DiskKVIdentity& id : f.store->live_identities()) {
        if (id.frontier != 0) { out.push_back(id.frontier); }
    }
    return out;
}

DiskKVBridgeStats DiskKVBridge::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

std::size_t DiskKVBridge::used_bytes(DiskKVKind kind) const {
    if (kind_index(kind) >= std::size(families_)) { return 0; }
    const Family& f = family(kind);
    return f.store ? f.store->used_bytes() : 0;
}

std::uint32_t DiskKVBridge::slot_count(DiskKVKind kind) const {
    if (kind_index(kind) >= std::size(families_)) { return 0; }
    const Family& f = family(kind);
    return f.store ? f.store->slot_count() : 0;
}

const std::string& DiskKVBridge::path(DiskKVKind kind) const {
    static const std::string kEmpty;
    if (kind_index(kind) >= std::size(families_)) { return kEmpty; }
    const Family& f = family(kind);
    return f.store ? f.path : kEmpty;
}

DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) {
    return families_[kind_index(kind)];
}
const DiskKVBridge::Family& DiskKVBridge::family(DiskKVKind kind) const {
    return families_[kind_index(kind)];
}

} // namespace ninfer
