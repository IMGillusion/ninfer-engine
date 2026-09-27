// Linux CPU-only: deterministic blocked flush + three-worker write stress.
#include <bits/stdc++.h>
#define private public
#include "core/disk_kv_bridge.h"
#undef private
using namespace ninfer;
using Clock = std::chrono::steady_clock;
static void until(const std::function<bool()>& done) {
    const auto end = Clock::now() + std::chrono::seconds(30);
    while (!done()) { assert(Clock::now() < end); std::this_thread::yield(); }
}
static DiskKVIdentity key(unsigned i) { return {.lo=i+1, .hi=71, .tag=92, .frontier=64}; }
static SpillTicket submit(DiskKVBridge& b, unsigned i, DiskKVKind kind,
                          const std::vector<std::byte>& bytes) {
    SpillTicket t;
    until([&] { t=b.try_submit(key(i),kind,bytes); return bool(t); });
    return t;
}
static void written(const SpillTicket& t) {
    until([&] { return DiskKVBridge::poll(t)!=SpillStatus::Pending; });
    assert(DiskKVBridge::poll(t)==SpillStatus::Written);
}
static void clean(DiskKVBridge& b) {
    b.wait_idle();
    { std::lock_guard lock(b.qmu_);
      assert(b.queue_.empty() && b.in_flight_==0 && b.flush_pending_==0); }
    for (auto& family:b.families_) {
        std::lock_guard lock(family.store->mu_);
        assert(!family.store->index_dirty_);
    }
}
int main() {
    const std::string root="/tmp/ninfer-flush-concurrency-test";
    std::filesystem::remove_all(root);
    DiskKVBridge::Options opts{.base_path=root,.main_page_stride=4096,
        .backend_page_stride=4096,.state_page_stride=4096,.capacity_bytes=64*1024*1024};
    std::vector<std::byte> bytes(4096,std::byte{73}), dst(4096);
    std::vector<SpillTicket> retained;
    {
        DiskKVBridge b(opts);
        assert(b.workers_.size()==3);
        // Hold a DIFFERENT family's store lock: the main write completes but
        // its idle flush blocks. Queue admission and another worker must work.
        for (unsigned round=0;round<12;++round) {
            std::unique_lock blocked(b.families_[1].store->mu_);
            written(submit(b,10000+round*2,DiskKVKind::MainKV,bytes));
            until([&] { std::lock_guard lock(b.qmu_);
                return b.flush_pending_==0 && b.in_flight_!=0; });
            auto idle=std::async(std::launch::async,[&] { b.wait_idle(); });
            assert(idle.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout);
            // Arrives after the first flush claimed its batch. Its increment
            // must survive the earlier flush and get its own idle publication.
            written(submit(b,10001+round*2,DiskKVKind::MainKV,bytes));
            assert(idle.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout);
            blocked.unlock();
            assert(idle.wait_for(std::chrono::seconds(30))==std::future_status::ready);
            idle.get(); clean(b);
        }
        // Six concurrent producers, three worker threads, every family. Bursts
        // cross the 64-write threshold repeatedly; no fixture eviction.
        std::vector<std::thread> producers;
        for (unsigned p=0;p<6;++p) producers.emplace_back([&,p] {
            std::vector<SpillTicket> tickets;
            for (unsigned i=0;i<256;++i)
                tickets.push_back(submit(b,p*256+i,static_cast<DiskKVKind>(p%3),bytes));
            for (const auto& t:tickets) written(t);
        });
        for (auto& p:producers) p.join();
        clean(b);
        assert(b.stats().spills==1560 && b.stats().queue_drops==0);
        // No wait_idle: destructor must finish accepted writes and final flush.
        for (unsigned i=0;i<96;++i)
            retained.push_back(submit(b,20000+i,static_cast<DiskKVKind>(i%3),bytes));
    }
    for (const auto& t:retained) written(t);
    {
        DiskKVBridge b(opts);
        for (unsigned p=0;p<6;++p) for (unsigned i=0;i<256;++i)
            assert(b.restore_page(key(p*256+i),static_cast<DiskKVKind>(p%3),dst) && dst==bytes);
        for (unsigned i=0;i<24;++i)
            assert(b.restore_page(key(10000+i),DiskKVKind::MainKV,dst) && dst==bytes);
        for (unsigned i=0;i<96;++i)
            assert(b.restore_page(key(20000+i),static_cast<DiskKVKind>(i%3),dst) && dst==bytes);
    }
    std::filesystem::remove_all(root);
    std::puts("PASS flush concurrency: 12 blocked-idle rounds, 6 producers/3 workers, 1656 writes reopened, destructor drain");
}
