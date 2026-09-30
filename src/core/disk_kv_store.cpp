#include "core/disk_kv_store.h"

#include "core/disk_kv_log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace ninfer {
namespace {

// NINFER_BRIDGE_TRACE=1 logs store write milestones for stall forensics.
inline bool store_trace() {
    static const bool on = std::getenv("NINFER_BRIDGE_TRACE") != nullptr;
    return on;
}

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kTrailerSize = 128;

std::size_t round_up(std::size_t v, std::size_t a) { return (v + (a - 1)) & ~(a - 1); }

std::string index_path(const std::string& data_path) {
    return data_path + ".idx";
}

struct CrcTable {
    // Slicing-by-8 tables for the SAME polynomial as the original byte-wise
    // table (0xEDB88320, the reflected CRC-32 / zlib polynomial) — values are
    // bit-identical to the byte-wise implementation, so every page ever
    // written stays valid. (The SSE4.2 _mm_crc32_* instruction computes the
    // *Castagnoli* CRC32C and is NOT value-compatible with this format.)
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    CrcTable() {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[0][i] = c;
        }
        for (std::size_t s = 1; s < 8; ++s) {
            for (std::uint32_t i = 0; i < 256; ++i) {
                const std::uint32_t prev = t[s - 1][i];
                t[s][i] = (prev >> 8) ^ t[0][prev & 0xFFu];
            }
        }
    }
};
const CrcTable& crc_table() {
    static const CrcTable tbl;
    return tbl;
}

} // namespace

std::uint64_t DiskKVStore::crc32c(std::span<const std::byte> data) {
    const auto& tbl = crc_table();
    const std::byte* p = data.data();
    std::size_t n = data.size();
    std::uint32_t c = 0xFFFFFFFFu;
    // Slicing-by-8: identical values to the original byte-wise loop, ~8-16x
    // faster (the byte-wise loop was the dominant cost of large sweeps).
    auto le32 = [](const std::byte* q) {
        return static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16) |
               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[3])) << 24);
    };
    while (n >= 8) {
        c ^= le32(p);
        const std::uint32_t next = le32(p + 4);
        c = tbl.t[7][c & 0xFFu] ^ tbl.t[6][(c >> 8) & 0xFFu] ^ tbl.t[5][(c >> 16) & 0xFFu] ^
            tbl.t[4][(c >> 24) & 0xFFu] ^ tbl.t[3][next & 0xFFu] ^
            tbl.t[2][(next >> 8) & 0xFFu] ^ tbl.t[1][(next >> 16) & 0xFFu] ^
            tbl.t[0][(next >> 24) & 0xFFu];
        p += 8;
        n -= 8;
    }
    for (; n > 0; --n, ++p) {
        const std::uint8_t v = static_cast<std::uint8_t>(*p);
        c = tbl.t[0][(c ^ v) & 0xFFu] ^ (c >> 8);
    }
    return static_cast<std::uint64_t>(c ^ 0xFFFFFFFFu);
}

// ---------- layout ----------

std::size_t DiskKVStore::slot_bytes() const noexcept {
    return round_up(kSlotHeaderSize + opts_.slot_size, kAlignment);
}
std::size_t DiskKVStore::data_bytes() const noexcept {
    return static_cast<std::size_t>(max_slots_) * slot_bytes();
}
std::size_t DiskKVStore::trailer_off() const noexcept {
    return round_up(data_bytes(), kAlignment);
}
std::size_t DiskKVStore::page_off(std::uint32_t slot) const noexcept {
    return static_cast<std::size_t>(slot) * slot_bytes() + kSlotHeaderSize;
}
DiskKVStore::Header* DiskKVStore::slot_hdr(std::uint32_t slot) const {
    return reinterpret_cast<Header*>(base_ + static_cast<std::size_t>(slot) * slot_bytes());
}
const DiskKVStore::Header* DiskKVStore::slot_hdr_c(std::uint32_t slot) const {
    return reinterpret_cast<const Header*>(base_ + static_cast<std::size_t>(slot) * slot_bytes());
}

// ---------- lifecycle ----------

DiskKVStore::DiskKVStore(Options opts) : opts_(std::move(opts)) {
    if (opts_.path.empty()) { throw std::invalid_argument("disk kv store path is empty"); }
    if (opts_.slot_size < kSlotHeaderSize) {
        throw std::invalid_argument("disk kv store slot_size too small");
    }
    if (opts_.max_slots == 0) {
        if (opts_.capacity_bytes == 0) {
            throw std::invalid_argument("disk kv store needs max_slots or capacity_bytes");
        }
        opts_.max_slots = static_cast<std::uint32_t>(
            opts_.capacity_bytes / round_up(kSlotHeaderSize + opts_.slot_size, kAlignment));
        if (opts_.max_slots == 0) { throw std::invalid_argument("capacity too small"); }
    }
    max_slots_ = opts_.max_slots;
    readers_.assign(max_slots_, 0);
    slot_pitch_ = slot_bytes();
    file_bytes_ = trailer_off() + kTrailerSize;

    struct stat st{};
    const bool data_exists = (::stat(opts_.path.c_str(), &st) == 0 && st.st_size > 0);
    if (!data_exists) {
        create_fresh();
    } else {
        if (static_cast<std::size_t>(st.st_size) < file_bytes_) {
            throw std::runtime_error("disk kv store " + opts_.path +
                                     " smaller than required by current geometry");
        }
        // Open the EXISTING file: never truncate.
        fd_ = ::open(opts_.path.c_str(), O_RDWR, 0644);
        if (fd_ < 0) {
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
        }
        void* mapped = ::mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            const int e = errno;
            ::close(fd_); fd_ = -1;
            throw std::runtime_error(std::string("mmap failed: ") + std::strerror(e));
        }
        base_ = static_cast<std::byte*>(mapped);
    }
    for (std::uint32_t s = 0; s < max_slots_; ++s) { free_slots_.push_back(s); }

    // Fast path: load the packed index (a full slot scan here costs minutes on
    // virtiofs-mounted stores; stale rows are checked against slot identities,
    // including interrupted batches). Data CRCs stay lazily verified at
    // read time by design (a corrupted page becomes a restore miss). Without a
    // valid index file a full scan rebuilds it.
    if (!load_index()) {
        rebuild_from_scan();
    }
}

void DiskKVStore::create_fresh() {
    fd_ = ::open(opts_.path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    }
    if (::ftruncate(fd_, static_cast<off_t>(file_bytes_)) != 0) {
        const int e = errno;
        ::close(fd_); fd_ = -1;
        throw std::runtime_error(std::string("ftruncate failed: ") + std::strerror(e));
    }
    void* mapped = ::mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
        const int e = errno;
        ::close(fd_); fd_ = -1;
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(e));
    }
    base_ = static_cast<std::byte*>(mapped);
}

DiskKVStore::~DiskKVStore() {
    if (base_ != nullptr) {
        ::munmap(base_, file_bytes_);
        base_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

// ---------- index persistence ----------

bool DiskKVStore::load_index() {
    const std::string path = index_path(opts_.path);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }  // no index yet: caller rebuilds
    IdxHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic != IdxHeader::kMagic ||
        hdr.version != 1 || hdr.count > max_slots_ ||
        hdr.slot_size != static_cast<std::uint32_t>(slot_bytes()) ||
        hdr.max_slots != max_slots_) {
        std::fclose(f);
        return false;  // foreign/corrupt index: caller rebuilds
    }
    std::vector<IdxEntry> rows(hdr.count);
    if (hdr.count > 0 &&
        std::fread(rows.data(), sizeof(IdxEntry), hdr.count, f) != hdr.count) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    index_.clear();
    index_.reserve(hdr.count);
    free_slots_.clear();
    free_slots_.reserve(max_slots_);

    // Header-only check per row: the index row must agree with the slot
    // header. If the slot was repurposed (different identity), drop the row.
    // Data integrity is verified lazily at READ time (read_page CRC) — a
    // corrupted page becomes a restore miss, not a startup cost.
    //
    // The header is pread from a small thread pool, NOT read through the
    // mapping: one random 4 KiB mapping fault per live page serialized this
    // loop into ~10 minutes at startup on the production volume (224K live
    // rows and growing with the store, 83% CPU in fault churn), while the
    // same reads through pread run in parallel. read_page_result still
    // verifies identity and CRC on every restore, so this pass guards only
    // the index's agreement with the data file — never data integrity.
    struct HeaderCheck {
        Header header{};
        bool ok = false;
    };
    std::vector<IdxEntry> candidates;
    candidates.reserve(rows.size());
    for (const IdxEntry& r : rows) {
        if (r.slot < max_slots_) { candidates.push_back(r); }  // defensive: ignore bad row
    }
    std::vector<HeaderCheck> checks(candidates.size());
    const auto validate_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const IdxEntry& r = candidates[i];
            Header h{};
            std::size_t got = 0;
            bool io_ok = true;
            while (got < sizeof(Header)) {
                const ssize_t n = ::pread(fd_, reinterpret_cast<char*>(&h) + got,
                                          sizeof(Header) - got,
                                          static_cast<off_t>(r.slot * slot_pitch_ + got));
                if (n <= 0) {
                    if (n < 0 && errno == EINTR) { continue; }
                    io_ok = false;
                    break;
                }
                got += static_cast<std::size_t>(n);
            }
            if (!io_ok) { continue; }  // unreadable header: drop the row
            const DiskKVIdentity id{r.lo, r.hi, r.tag, r.frontier};
            checks[i].ok = h.magic == Header::kMagic && h.lo == id.lo &&
                           h.hi == id.hi && h.tag == id.tag && h.frontier == id.frontier;
            if (checks[i].ok) { checks[i].header = h; }
        }
    };
    const auto load_started = std::chrono::steady_clock::now();
    const std::size_t total = candidates.size();
    if (total != 0) {
        unsigned workers = std::thread::hardware_concurrency();
        workers = std::clamp(workers == 0 ? 4U : workers, 2U, 8U);
        struct Range {
            std::size_t begin = 0;
            std::size_t end = 0;
        };
        std::vector<Range> ranges;
        for (unsigned t = 0; t < workers; ++t) {
            const std::size_t b = total * t / workers;
            const std::size_t e = total * (t + 1) / workers;
            if (b < e) { ranges.push_back(Range{b, e}); }
        }
        std::vector<std::thread> pool;
        std::size_t spawned = 1;  // range 0 always runs on this thread
        try {
            for (std::size_t i = 1; i < ranges.size(); ++i) {
                pool.emplace_back(validate_range, ranges[i].begin, ranges[i].end);
                spawned = i + 1;
            }
        } catch (...) {
            // A smaller pool than planned is fine; unspawned ranges run below.
        }
        validate_range(ranges[0].begin, ranges[0].end);
        for (auto& worker : pool) { worker.join(); }
        for (std::size_t i = spawned; i < ranges.size(); ++i) {
            validate_range(ranges[i].begin, ranges[i].end);
        }
    }
    std::size_t dropped = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!checks[i].ok) { ++dropped; continue; }
        // Rebuild the LRU map as well: a restart must never leave the
        // eviction index empty. Observed failure: the fast path filled
        // index_ but not lru_, so a full store failed EVERY upsert with
        // "no victim" after redeploy (208 FAILs / 19 owner-retained 503s
        // in one window) — evict_one_lru had no candidates at all.
        const DiskKVIdentity id{candidates[i].lo, candidates[i].hi,
                                candidates[i].tag, candidates[i].frontier};
        record_live(id, candidates[i].slot, checks[i].header.last_used);
    }
    disk_kv_logf('I', "l3-store",
                 "index fast-path | rows=%zu dropped=%zu took=%.1fs",
                 candidates.size(), dropped,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count());

    // Rebuild the free pool from the (verified) live slots.
    std::vector<bool> live(max_slots_, false);
    for (const auto& [id, s] : index_) { live[s] = true; }
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        if (!live[s]) { free_slots_.push_back(s); }
    }
    return true;
}

void DiskKVStore::rebuild_from_scan() {
    index_.clear();
    readers_.assign(max_slots_, 0);
    free_slots_.clear();
    free_slots_.reserve(max_slots_);
    clock_ = 0;
    std::vector<bool> live(max_slots_, false);
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        const Header& h = *slot_hdr_c(s);
        if (h.magic != Header::kMagic) { continue; }
        const std::uint64_t crc =
            crc32c(std::span<const std::byte>(base_ + page_off(s), opts_.slot_size));
        if (crc != h.crc) { continue; }  // torn/corrupt slot: dead
        const DiskKVIdentity id{h.lo, h.hi, h.tag, h.frontier};
        record_live(id, s, h.last_used);  // index_ + lru_ (see load_index note)
        live[s] = true;
    }
    for (std::uint32_t s = 0; s < max_slots_; ++s) {
        if (!live[s]) { free_slots_.push_back(s); }
    }
    rebuilt_from_scan_ = true;
    index_dirty_ = true;
    persist_index_unlocked();
}

void DiskKVStore::persist_index_unlocked() {
    if (!index_dirty_) { return; }
    // Atomic publication only: no fsync on the hot path.
    const std::string path = index_path(opts_.path);
    const std::string tmp  = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { return; }
    IdxHeader hdr{};
    hdr.magic     = IdxHeader::kMagic;
    hdr.version   = 1;
    hdr.count     = static_cast<std::uint32_t>(index_.size());
    hdr.slot_size = static_cast<std::uint32_t>(slot_bytes());
    hdr.max_slots = max_slots_;
    hdr.clock     = clock_;
    bool ok = std::fwrite(&hdr, sizeof(hdr), 1, f) == 1;
    for (const auto& [id, s] : index_) {
        IdxEntry r{};
        r.lo = id.lo;
        r.hi = id.hi;
        r.tag = id.tag;
        r.frontier = id.frontier;
        r.slot = s;
        if (std::fwrite(&r, sizeof(r), 1, f) != 1) { ok = false; break; }
    }
    if (std::fflush(f) != 0) { ok = false; }
    if (std::fclose(f) != 0) { ok = false; }
    // Deliberately NO fsync on the hot path: a torn index is recovered by a
    // full header rescan on next open (cheap: slots are CRC-verified), and
    // the spill worker would otherwise stall behind fsync latency per page.
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        return;
    }
    index_dirty_ = false;
}

// ---------- internals ----------

std::uint64_t DiskKVStore::bump_clock() noexcept {
    return ++clock_;
}

void DiskKVStore::zero_slot(std::uint32_t slot) {
    std::memset(slot_hdr(slot), 0, sizeof(Header));
}

void DiskKVStore::record_live(const DiskKVIdentity& id, std::uint32_t slot, std::uint64_t last_used) {
    index_[id] = slot;
    if (last_used > clock_) { clock_ = last_used; }
    lru_insert(slot, last_used);
}

void DiskKVStore::lru_insert(std::uint32_t slot, std::uint64_t last_used) {
    // Caller holds mu_.
    lru_erase(slot);
    auto it = lru_.emplace(last_used, slot).first;
    lru_slot_[slot] = it;
}

void DiskKVStore::lru_erase(std::uint32_t slot) noexcept {
    // Caller holds mu_.
    const auto sit = lru_slot_.find(slot);
    if (sit != lru_slot_.end()) {
        lru_.erase(sit->second);
        lru_slot_.erase(sit);
    }
}

void DiskKVStore::release_slot(std::uint32_t slot) {
    free_slots_.push_back(slot);
}

std::optional<DiskKVStore::EvictedPage> DiskKVStore::evict_one_lru() {
    // Caller holds mu_. The LRU map yields the coldest eligible slot in
    // O(log n) amortized; a protected or mid-read victim is skipped. (The
    // previous full-index scan touched every slot's mmap header page — ~26K
    // faults per scan on a store larger than VM memory — and held mu_ for
    // minutes once the store was full, starving every other store op.)
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        const std::uint32_t s = it->second;
        if (s >= readers_.size()) { continue; }
        if (readers_[s] != 0) { continue; }                  // mid-read: do not recycle
        const Header& h = *slot_hdr_c(s);
        if (h.magic != Header::kMagic) { continue; }         // stale slot: skip
        const DiskKVIdentity vid{h.lo, h.hi, h.tag, h.frontier};
        if (protected_identity(vid)) { continue; }
        const EvictedPage victim{vid, s};
        lru_.erase(it);
        lru_slot_.erase(s);
        index_.erase(vid);
        zero_slot(s);
        release_slot(s);
        return victim;
    }
    return std::nullopt;
}

bool DiskKVStore::identity_matches(const Header& h, const DiskKVIdentity& id) const {
    return h.lo == id.lo && h.hi == id.hi && h.tag == id.tag && h.frontier == id.frontier;
}

// ---------- API ----------

std::uint32_t DiskKVStore::live_slots() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<std::uint32_t>(index_.size());
}
std::size_t DiskKVStore::used_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.size() * slot_pitch_;
}
std::size_t DiskKVStore::free_bytes() const noexcept {
    return static_cast<std::size_t>(max_slots_ - live_slots()) * slot_pitch_;
}

bool DiskKVStore::upsert_page(const DiskKVIdentity& id, std::span<const std::byte> bytes,
                             std::vector<std::uint32_t>* evicted) {
    if (bytes.size() != opts_.slot_size || max_slots_ == 0) { return false; }
    const bool trace = store_trace();
    const auto t0 = std::chrono::steady_clock::now();
    if (trace) {
        disk_kv_logf('D', "l3-store", "upsert enter | id=%llu/%llu",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi));
    }
    std::uint32_t slot = 0;
    std::uint64_t stamp = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);

        const auto it = index_.find(id);
        if (it != index_.end()) {
            // Already restorable: refresh LRU in memory (flushed by flush_index).
            const auto stamp2 = bump_clock();
            (*slot_hdr(it->second)).last_used = stamp2;
            lru_insert(it->second, stamp2);
            index_dirty_ = true;
            return true;
        }

        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            // Full: LRU-evict the coldest page.
            auto victim = evict_one_lru();
            if (!victim.has_value()) {
                disk_kv_logf('E', "l3-store", "upsert EVICT-FAIL no-victim | id=%llu/%llu",
                             static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi));
                return false;
            }
            if (evicted != nullptr) {
                evicted->push_back(victim->slot);
            }
            // evict_one_lru() already released the victim slot into free_slots_.
            slot = free_slots_.back();
            free_slots_.pop_back();
        }
        stamp = bump_clock();
        // Clear the previous (now orphaned) header before the unlocked write:
        // a stale magic must never validate against half-written data.
        zero_slot(slot);
    }
    if (trace) {
        disk_kv_logf('D', "l3-store", "upsert stage-mu1 | id=%llu/%llu | took=%.3fs",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // ---- Unlocked region: the slot is off free_slots_ and absent from
    // index_, so no reader, probe or LRU evictor can observe or reclaim it.
    // The header is stamped last (under the lock) so the page becomes
    // reachable only once its data and CRC are complete. Parallel writers
    // therefore only contend on the short bookkeeping sections.
    const std::uint32_t crc =
        static_cast<std::uint32_t>(crc32c(std::span<const std::byte>(bytes.data(),
                                                                    opts_.slot_size)));
    // Payload via pwrite, not the mmap: the store file is far larger than the
    // VM's memory, so touching unmapped mapping pages here faults ~288 times
    // per 1.18MB page (4KB granularity) and the bridge workers stall for
    // minutes while the boundary spins (observed: 144 upserts entered, 0
    // completed, spills stuck at 0 for 6+ minutes).
    {
        const std::size_t payload_off = page_off(slot);
        std::size_t written = 0;
        while (written < opts_.slot_size) {
            const ssize_t n = ::pwrite(fd_, bytes.data() + written,
                                       opts_.slot_size - written,
                                       static_cast<off_t>(payload_off + written));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) { continue; }
                disk_kv_logf('E', "l3-store", "upsert PWRITE-FAIL | errno=%d written=%zu/%zu | id=%llu/%llu",
                             errno, written, opts_.slot_size,
                             static_cast<unsigned long long>(id.lo),
                             static_cast<unsigned long long>(id.hi));
                // The slot is off free_slots_ and absent from index_ (its header
                // was zeroed), so returning it here loses nothing: the store is
                // full-capacity again instead of leaking one slot per I/O error.
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    release_slot(slot);
                }
                return false;
            }
            written += static_cast<std::size_t>(n);
        }
    }
    if (trace) {
        disk_kv_logf('D', "l3-store", "upsert stage-write | id=%llu/%llu | took=%.3fs",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    if (trace) {
        disk_kv_logf('D', "l3-store", "upsert stage-mu2-enter | id=%llu/%llu | took=%.3fs",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (trace) {
            disk_kv_logf('D', "l3-store", "upsert stage-mu2-got | id=%llu/%llu | took=%.3fs",
                         static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        if (index_.find(id) != index_.end()) {
            // Lost a race with another writer of the same id (both passed the
            // pre-check): the id is written; return our slot instead of
            // leaking it or double-linking.
            free_slots_.push_back(slot);
            return true;
        }
        Header nh{};
        nh.magic     = Header::kMagic;
        nh.lo        = id.lo;
        nh.hi        = id.hi;
        nh.tag       = id.tag;
        nh.frontier  = id.frontier;
        nh.crc       = crc;
        nh.last_used = stamp;
        std::memcpy(slot_hdr(slot), &nh, sizeof(Header));
        record_live(id, slot, nh.last_used);
        index_dirty_ = true;
        // Preserve eager publication for direct users; batch workers opt in.
        if (!opts_.defer_index_updates) { persist_index_unlocked(); }
    }
    if (trace) {
        disk_kv_logf('D', "l3-store", "upsert done | id=%llu/%llu | took=%.3fs",
                     static_cast<unsigned long long>(id.lo), static_cast<unsigned long long>(id.hi),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return true;
}

bool DiskKVStore::read_page(const DiskKVIdentity& id, std::span<std::byte> dst) {
    return read_page_result(id, dst) == DiskReadResult::Success;
}

DiskReadResult DiskKVStore::read_page_result(const DiskKVIdentity& id, std::span<std::byte> dst) {
    if (dst.size() != opts_.slot_size) { return DiskReadResult::BadSize; }
    std::size_t off = 0;
    std::uint32_t slot = 0;
    std::uint64_t expected_crc = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = index_.find(id);
        if (it == index_.end()) { return DiskReadResult::Missing; }
        slot = it->second;
        if (slot >= readers_.size()) { return DiskReadResult::BadHeader; }
        const Header& h = *slot_hdr_c(slot);
        if (h.magic != Header::kMagic) { return DiskReadResult::BadHeader; }
        if (!identity_matches(h, id)) { return DiskReadResult::BadIdentity; }
        expected_crc = h.crc;
        off = page_off(slot);
        ++readers_[slot];                    // pin: eviction must not recycle it
        const auto stamp2 = bump_clock();
        (*slot_hdr(slot)).last_used = stamp2;
        lru_insert(slot, stamp2);
        index_dirty_ = true;
    }
    // Payload validation and copy happen OUTSIDE the store lock so concurrent
    // readers actually run in parallel (see readers_ in the header). Read via
    // pread, not the mmap: the store file is far larger than the VM's memory
    // and mmap reads fault-storm the same way writes do (see upsert_page).
    bool io_ok = true;
    {
        std::size_t got = 0;
        while (got < opts_.slot_size) {
            const ssize_t n = ::pread(fd_, dst.data() + got, opts_.slot_size - got,
                                      static_cast<off_t>(off + got));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) { continue; }
                io_ok = false;
                break;
            }
            got += static_cast<std::size_t>(n);
        }
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto& pinned = readers_[slot];
        if (pinned) { --pinned; }
    }
    // A pread failure is an I/O problem (disk, mount, short file), not data
    // corruption: report it as its own class so triage does not chase CRC ghosts.
    if (!io_ok) { return DiskReadResult::IOFailure; }
    if (opts_.verify_crc &&
        crc32c(std::span<const std::byte>(dst.data(), opts_.slot_size)) != expected_crc) {
        return DiskReadResult::BadCRC;
    }
    return DiskReadResult::Success;
}

bool DiskKVStore::contains(const DiskKVIdentity& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.find(id) != index_.end();
}

void DiskKVStore::flush_index() {
    std::lock_guard<std::mutex> lock(mu_);
    persist_index_unlocked();
}

bool DiskKVStore::touch(const DiskKVIdentity& id) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    const auto stamp = bump_clock();
    (*slot_hdr(it->second)).last_used = stamp;
    // The in-memory LRU map decides eviction; updating only the header would
    // leave the touched page at its old (cold) eviction position.
    lru_insert(it->second, stamp);
    index_dirty_ = true;
    return true;
}

bool DiskKVStore::touch_lru(const DiskKVIdentity& id) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    // In-memory only by design (see the header comment): no mmap header write
    // (each would cost one 4 KiB mapping fault — the sweep touches thousands
    // of pages per pass) and no index dirtying (nothing persistent changed).
    lru_insert(it->second, bump_clock());
    return true;
}

bool DiskKVStore::evict(const DiskKVIdentity& id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (protected_identity(id)) return false;
    const auto it = index_.find(id);
    if (it == index_.end()) { return false; }
    const std::uint32_t slot = it->second;
    if (slot < readers_.size() && readers_[slot] != 0) { return false; }   // mid-read
    index_.erase(it);
    zero_slot(slot);
    release_slot(slot);
    index_dirty_ = true;
    persist_index_unlocked();
    return true;
}

std::vector<DiskKVIdentity> DiskKVStore::evict_until_free(std::uint32_t free) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<DiskKVIdentity> gone;
    // live + free == capacity normally; requests above capacity saturate.
    while (free_slots_.size() < free && !index_.empty()) {
        auto v = evict_one_lru();
        if (!v.has_value()) { break; }
        gone.push_back(v->id);
    }
    if (!gone.empty()) {
        index_dirty_ = true;
        persist_index_unlocked();
    }
    return gone;
}

std::vector<DiskKVIdentity> DiskKVStore::live_identities() const {
    // Short-TTL cache (see the member comment): the admission probe hammers
    // this while a transaction stalls, and each full index copy under mu_
    // starved the bridge workers' mu2 sections.
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    if (now - live_cache_at_ < std::chrono::milliseconds(250)) { return live_cache_; }
    std::vector<DiskKVIdentity> out;
    out.reserve(index_.size());
    for (const auto& [id, s] : index_) { (void)s; out.push_back(id); }
    live_cache_ = out;
    live_cache_at_ = now;
    return out;
}

std::vector<std::uint32_t> DiskKVStore::live_slot_list() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::uint32_t> out;
    out.reserve(index_.size());
    for (const auto& [id, s] : index_) { out.push_back(s); }
    return out;
}

} // namespace ninfer
