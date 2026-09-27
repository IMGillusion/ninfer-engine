// Linux pure-CPU regression. White-box access only to hold store/queue locks
// deterministically: no timing-dependent assumptions about worker speed.
#include <bits/stdc++.h>
#define private public
#include "core/disk_kv_bridge.h"
#undef private
using namespace ninfer;
using Clock = std::chrono::steady_clock;
static void until(const std::function<bool()>& done) {
    const auto deadline = Clock::now() + std::chrono::seconds(20);
    while (!done()) {
        assert(Clock::now() < deadline);
        std::this_thread::yield();
    }
}
static ProbeBatchTicket submit(DiskKVBridge& b, std::span<const DiskKVIdentity> ids,
                              DiskKVKind kind = DiskKVKind::MainKV) {
    ProbeBatchTicket t;
    until([&] { t = b.try_probe_batch(ids, kind); return bool(t); });
    return t;
}
static void check(const ProbeBatchTicket& t, SpillStatus status) {
    until([&] { return DiskKVBridge::poll(t); });
    for (size_t i = 0; i < t->size; ++i) assert(t->results[i] == status);
}
int main() {
    const std::string root = "/tmp/ninfer-batch-probe-test";
    std::filesystem::remove_all(root);
    DiskKVBridge::Options opts{.base_path=root, .main_page_stride=4096,
        .backend_page_stride=4096, .state_page_stride=4096, .capacity_bytes=4*1024*1024};
    const DiskKVIdentity present{.lo=1, .hi=2, .tag=3, .frontier=64};
    const DiskKVIdentity missing{.lo=7, .hi=8, .tag=3, .frontier=128};
    std::vector<std::byte> bytes(4096, std::byte{42});
    std::array<DiskKVIdentity, 32> ids; ids.fill(present);
    std::vector<ProbeBatchTicket> retained;
    {
        DiskKVBridge b(opts);
        for (auto kind : {DiskKVKind::MainKV, DiskKVKind::BackendKV, DiskKVKind::StateImage}) {
            assert(b.spill_page_sync(present, kind, bytes));
            std::array input{present, missing, present};
            auto t = submit(b, input, kind);
            input.fill(missing); // queued identities must not borrow caller storage
            until([&] { return DiskKVBridge::poll(t); });
            assert(t->size == 3 && t->results[0] == SpillStatus::Present);
            assert(t->results[1] == SpillStatus::Missing && t->results[2] == SpillStatus::Present);
        }
        for (size_t count : {size_t(0), size_t(33)}) {
            std::vector<DiskKVIdentity> bad(count);
            bool threw = false;
            try { (void)b.try_probe_batch(bad, DiskKVKind::MainKV); }
            catch (const std::invalid_argument&) { threw = true; }
            assert(threw);
        }
        check(submit(b, ids, static_cast<DiskKVKind>(255)), SpillStatus::Failed);
        // Legacy single-page contract still works alongside the new overload.
        SpillTicket single;
        until([&] { single=b.try_probe(present, DiskKVKind::MainKV); return bool(single); });
        until([&] { return DiskKVBridge::poll(single) != SpillStatus::Pending; });
        assert(DiskKVBridge::poll(single) == SpillStatus::Present);
        b.wait_idle();
        // Queue mutex contention returns promptly without accepting partial work.
        {
            std::lock_guard lock(b.qmu_);
            auto result = std::async(std::launch::async, [&] {
                return b.try_probe_batch(ids, DiskKVKind::MainKV);
            });
            assert(result.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
            assert(!result.get());
        }
        // Stall all workers in contains(), outside qmu_, then reach EXACTLY the
        // existing 512-job (queued + in-flight) bound. Each job has 32 pages.
        {
            std::lock_guard store_lock(b.families_[0].store->mu_);
            for (size_t i=0; i<b.kQueueCap; ++i) retained.push_back(submit(b, ids));
            assert(!b.try_probe_batch(ids, DiskKVKind::MainKV));
            std::lock_guard queue_lock(b.qmu_);
            assert(b.queue_.size() + b.in_flight_ == b.kQueueCap);
            assert(b.stats().queue_drops == 0);
            ids.fill(missing); // all workers blocked: proves owned identity copy
        }
        for (auto& t : retained) check(t, SpillStatus::Present);
        ids.fill(present);
        // Retry rejected batch unchanged; every page resolves, no drop counter.
        check(submit(b, ids), SpillStatus::Present);
        assert(b.stats().queue_drops == 0);
        // Concurrent producers: worker ownership and publication stress.
        std::vector<std::thread> producers;
        for (int p=0; p<4; ++p) producers.emplace_back([&] {
            for (int i=0; i<128; ++i) check(submit(b, ids), SpillStatus::Present);
        });
        for (auto& p : producers) p.join();
        b.wait_idle();
        // Accepted work completes even if caller abandons its ticket.
        (void)submit(b, ids);
        // No wait_idle before destruction: destructor must drain accepted jobs.
        for (int i=0; i<128; ++i) retained.push_back(submit(b, ids));
    }
    for (auto& t : retained) check(t, SpillStatus::Present);
    {
        DiskKVBridge b(opts);
        check(submit(b, ids), SpillStatus::Present);
    }
    {
        DiskKVBridge disabled({});
        check(submit(disabled, ids), SpillStatus::Failed);
    }
    {
        opts.base_path = root + "/main-only";
        opts.backend_page_stride = opts.state_page_stride = 0;
        DiskKVBridge b(opts);
        check(submit(b, ids, DiskKVKind::BackendKV), SpillStatus::Failed);
        check(submit(b, ids, DiskKVKind::StateImage), SpillStatus::Failed);
        ids.fill(missing);
        check(submit(b, ids), SpillStatus::Missing);
    }
    std::filesystem::remove_all(root);
    std::puts("PASS batch probe: ordered mixed/duplicates/3 families/copied input/invalid input/Failed/legacy; 512x32 saturation+retry, 4x128 concurrent batches, destructor drain/reopen, zero drops");
}
