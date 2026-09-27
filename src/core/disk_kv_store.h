#pragma once
// L3 (disk) tier for the KV cache hierarchy — persistent, self-describing,
// true-LRU.
//
// NInfer's cache is two levels: a device page pool (kv-capacity tokens) and a
// pinned host arena (--host-kv-mib). When the host arena fills, the pressure
// planner used to DROP the affected checkpoints permanently, so the next turn
// of that session paid full prefill recompute. This store is a cold,
// file-backed third level: a host-full event demotes a page run to disk, and
// a later turn restores it (disk read + H2D) instead of recomputing.
//
// PERSISTENCE MODEL (engine restarts happen — watchdog + manual; data that
// dies with the process is worth nothing):
//   * DATA file: 4KiB-aligned slots, each with a 48-byte SELF-DESCRIBING
//     header (magic + content identity + CRC32C + LRU timestamp). Slot
//     placement is managed by the store (free pool), NOT by a hash of the
//     identity — identity->slot lives in the index.
//   * INDEX file (<path>.idx): the identity->slot mapping, rewritten
//     atomically (tmp + rename) whenever the live set changes by default.
//     No power-loss durability guarantee (no fsync/msync).
//   * Open: valid index -> fast path. Missing/corrupt index -> rebuild by
//     scanning slot headers (slow, one-time; CRC-verified).
//
// EVICTION: true LRU over the live set (lowest last_used first). Reads and
// explicit touches refresh last_used, so an active session's pages outlive
// the coldest sessions'.
//
// Self-contained (no CUDA, no engine types): builds and unit-tests in seconds.

#include <cstddef>
#include <atomic>
#include <chrono>
#include <memory>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <vector>

namespace ninfer {

/** Content identity of one KV page (mirrors the engine's prefix digest). */
struct DiskKVIdentity {
    std::uint64_t lo       = 0;  // rolling digest of the prefix at the page end
    std::uint64_t hi       = 0;
    std::uint32_t tag      = 0;  // engine identity_tag (spec backend | proposal<<8 | dtype<<16)
    std::uint32_t frontier = 0;  // page end token position (observability)

    [[nodiscard]] bool operator==(const DiskKVIdentity&) const noexcept = default;
};

struct DiskKVIdentityHash {
    std::size_t operator()(const DiskKVIdentity& id) const noexcept {
        std::uint64_t h = id.lo;
        h ^= id.hi + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::uint64_t(id.tag) * 0xff51afd7ed558ccdULL;
        h ^= std::uint64_t(id.frontier) * 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 31;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }
};

enum class DiskReadResult { Pending, Success, Missing, BadHeader, BadIdentity, BadCRC, BadSize, IOFailure, Disabled };

class DiskKVStore {
public:
    struct Options {
        std::string path;         // data file (created if absent; NEVER truncated)
        std::size_t slot_size    = 0;   // bytes per KV page (engine page stride)
        std::size_t capacity_bytes = 0; // 0 = invalid (one of the two is required)
        std::uint32_t max_slots   = 0;  // explicit capacity (preferred)
        bool verify_crc          = true;
        // Opt-in for workers calling flush_index() at batch/idle/shutdown.
        // Defers new-page publication ONLY; explicit evictions stay eager.
        // Crash/close before flush can lose new pages, never trust stale rows.
        // Destructor does not flush; owner must flush before shutdown.
        bool defer_index_updates = false;
    };

    /** Open the backing file and the identity index. Throws on I/O errors or
     *  a data file laid out for a different geometry. */
    explicit DiskKVStore(Options opts);
    ~DiskKVStore();

    DiskKVStore(const DiskKVStore&)            = delete;
    DiskKVStore& operator=(const DiskKVStore&) = delete;

    [[nodiscard]] std::size_t    slot_size()  const noexcept { return opts_.slot_size; }
    [[nodiscard]] std::uint32_t  slot_count() const noexcept { return max_slots_; }
    [[nodiscard]] std::uint32_t  live_slots() const noexcept;
    [[nodiscard]] std::size_t    used_bytes() const noexcept;
    [[nodiscard]] std::size_t    free_bytes() const noexcept;
    [[nodiscard]] const std::string& path() const noexcept { return opts_.path; }
    [[nodiscard]] bool            index_rebuilt_from_scan() const noexcept { return rebuilt_from_scan_; }

    /** Insert (or refresh) the page. When the store is full, evicts the
     *  LRU page first (its id reported via `evicted`, may be null).
     *  Returns false only when the store is full and no slot was evictable. */
    bool upsert_page(const DiskKVIdentity& id, std::span<const std::byte> bytes,
                     std::vector<std::uint32_t>* evicted = nullptr);

    /** Read the page into `dst` (slot_size bytes). Identity + CRC verified;
     *  refreshes the LRU timestamp. */
    bool read_page(const DiskKVIdentity& id, std::span<std::byte> dst);
    DiskReadResult read_page_result(const DiskKVIdentity& id, std::span<std::byte> dst);

    /** Presence check (cheap; no data read, no LRU refresh). */
    [[nodiscard]] bool contains(const DiskKVIdentity& id) const;

    /** Refresh the slot's LRU timestamp without reading the page. */
    bool touch(const DiskKVIdentity& id);

    /** Evict a specific page (if it currently holds `id`). */
    bool evict(const DiskKVIdentity& id);

    /** LRU-evict until `free` slots are free. Returns evicted identities. */
    [[nodiscard]] std::vector<DiskKVIdentity> evict_until_free(std::uint32_t free);

    /** Identities of every live page in this store (read-side frontier scan:
     *  the engine matches its own digest chain against each live state
     *  frontier to find a restorable boundary). Bounded by max_slots. */
    [[nodiscard]] std::vector<DiskKVIdentity> live_identities() const;

    /** Slot indices currently live (diagnostics). */
    [[nodiscard]] std::vector<std::uint32_t> live_slot_list() const;

    /** Publish pending LRU/index changes (batch/idle/shutdown barrier).
     *  Atomic replacement, NOT an fsync/power-loss durability barrier. */
    void flush_index();

    using ProtectedKeys = std::unordered_set<DiskKVIdentity, DiskKVIdentityHash>;
    // Nonblocking, single maintenance lease. It protects future writes too.
    bool try_protect(std::shared_ptr<const ProtectedKeys> keys) noexcept {
        std::shared_ptr<const ProtectedKeys> empty;
        return std::atomic_compare_exchange_strong(&protected_keys_, &empty, std::move(keys));
    }
    void unprotect() noexcept { std::atomic_store(&protected_keys_, std::shared_ptr<const ProtectedKeys>{}); }

    static constexpr std::size_t kSlotHeaderSize = 48;
    static std::uint64_t crc32c(std::span<const std::byte> data);

private:
    struct Header {  // on-disk, 48 bytes, head of every slot
        std::uint32_t magic     = 0;
        std::uint32_t reserved  = 0;
        std::uint64_t last_used = 0;
        std::uint64_t lo        = 0;
        std::uint64_t hi        = 0;
        std::uint32_t tag       = 0;
        std::uint32_t frontier  = 0;
        std::uint32_t crc       = 0;
        std::uint32_t pad       = 0;
        static constexpr std::uint32_t kMagic = 0x4E44564B;  // 'NKVD'
    };

    struct IdxEntry {  // on-disk index row, 32 bytes
        std::uint64_t lo       = 0;
        std::uint64_t hi       = 0;
        std::uint32_t tag      = 0;
        std::uint32_t frontier = 0;
        std::uint32_t slot     = 0;
        std::uint32_t pad      = 0;
    };

    struct IdxHeader {  // on-disk index header, 32 bytes
        std::uint32_t magic   = 0;  // 'KDVI'
        std::uint32_t version = 0;
        std::uint32_t count   = 0;
        std::uint32_t slot_size = 0;
        std::uint32_t max_slots = 0;
        std::uint64_t clock   = 0;  // LRU timestamp base, survives restarts
        static constexpr std::uint32_t kMagic = 0x4944564B;
    };

    [[nodiscard]] std::size_t slot_bytes() const noexcept;
    [[nodiscard]] std::size_t data_bytes() const noexcept;
    [[nodiscard]] std::size_t trailer_off() const noexcept;
    [[nodiscard]] std::size_t page_off(std::uint32_t slot) const noexcept;
    [[nodiscard]] Header* slot_hdr(std::uint32_t slot) const;
    [[nodiscard]] const Header* slot_hdr_c(std::uint32_t slot) const;
    [[nodiscard]] bool identity_matches(const Header& h, const DiskKVIdentity& id) const;

    void create_fresh();
    bool load_index();                    // fast path; false -> caller rebuilds
    void rebuild_from_scan();             // slow path: scan all slot headers
    /** Rewrite the index file atomically (tmp+rename, no fsync on hot path).
     *  A torn index is safe: next open falls back to a header rescan. */
    void persist_index_unlocked();
    std::uint64_t bump_clock() noexcept;
    void zero_slot(std::uint32_t slot);
    void record_live(const DiskKVIdentity& id, std::uint32_t slot, std::uint64_t last_used);
    void lru_insert(std::uint32_t slot, std::uint64_t last_used);
    void lru_erase(std::uint32_t slot) noexcept;
    void release_slot(std::uint32_t slot);
    struct EvictedPage {
        DiskKVIdentity id;
        std::uint32_t slot;
    };
    /** LRU core: evict the lowest-last_used live slot. Caller holds mu_. */
    std::optional<EvictedPage> evict_one_lru();

    mutable std::mutex mu_;
    std::shared_ptr<const ProtectedKeys> protected_keys_;
    // O(log n) LRU index (last_used -> slot) plus slot->iterator. Replaces
    // evict_one_lru's O(n) full-index scan, which touched every slot's mmap
    // header page (~26K faults per scan on a store larger than VM memory)
    // while holding mu_ for minutes once the store was full.
    std::map<std::uint64_t, std::uint32_t> lru_;
    std::unordered_map<std::uint32_t, std::map<std::uint64_t, std::uint32_t>::iterator> lru_slot_;
    // Short-TTL cache of live_identities(): the admission probe calls it
    // thousands of times per second while a transaction stalls, and the full
    // index copy under mu_ starved the bridge workers' mu2 sections
    // (observed: 11 upserts never returned from mu2). 250ms staleness is fine
    // for candidate selection because the actual reads verify the pages.
    mutable std::vector<DiskKVIdentity> live_cache_;
    mutable std::chrono::steady_clock::time_point live_cache_at_{};
    bool protected_identity(const DiskKVIdentity& id) const noexcept {
        auto keys = std::atomic_load(&protected_keys_);
        if (!keys) return false;
        return keys->contains(id);
    }
    Options      opts_;
    int          fd_         = -1;
    std::byte*   base_       = nullptr;
    std::size_t  file_bytes_ = 0;
    std::size_t  slot_pitch_ = 0;
    std::uint32_t max_slots_ = 0;
    std::uint64_t clock_     = 0;
    bool         rebuilt_from_scan_ = false;
    bool         index_dirty_ = false; // protected by mu_ after construction

    std::unordered_map<DiskKVIdentity, std::uint32_t, DiskKVIdentityHash> index_;
    // Per-slot in-flight READ count (protected by mu_). Read payloads are
    // CRC-checked and copied OUTSIDE mu_: holding the lock across 1.18MB of
    // CRC+memcpy serialized every reader at ~1.2 ms/page (~980MB/s), while the
    // same NVMe sustains ~880MB/s with only four concurrent streams. A pinned
    // slot must not be recycled under a copy, so eviction skips readers_ != 0.
    std::vector<std::uint32_t> readers_;
    std::vector<std::uint32_t> free_slots_;
};

} // namespace ninfer
