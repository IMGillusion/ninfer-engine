#include "core/owner_spill_batch.h"
#include <cassert>
#include <iostream>
using namespace ninfer;
struct Fake {
    ProbeBatchTicket probe;
    std::vector<SpillTicket> writes;
    std::vector<unsigned> ids;
    DiskKVKind kind = DiskKVKind::MainKV;
    unsigned credit = 0, probes = 0;
    bool probe_busy = false;
    ProbeBatchTicket try_probe_batch(std::span<const DiskKVIdentity> keys, DiskKVKind k) {
        ++probes;
        if (probe_busy) return {};
        kind = k;
        probe = std::make_shared<ProbeBatchCompletion>(); probe->size = keys.size();
        return probe;
    }
    SpillTicket try_submit(const DiskKVIdentity& id, DiskKVKind k, std::span<const std::byte> bytes) {
        assert(k == kind && bytes.size() == 4);
        if (!credit) return {};
        --credit;
        ids.push_back(id.frontier);
        auto t = std::make_shared<SpillCompletion>(); writes.push_back(t); return t;
    }
};
int main() {
    unsigned scenarios = 0;
    for (auto kind : {DiskKVKind::MainKV, DiskKVKind::BackendKV, DiskKVKind::StateImage}) {
        OwnerSpillBatch b; Fake f; b.batch_kind = kind;
        for(unsigned i=0;i<32;++i) b.batch_ids.push_back({.frontier=i});
        unsigned prepared=0, saved=0, dedup=0;
        std::byte bytes[4]{};
        auto prepare=[&](const auto& missing, auto& payloads) {
            ++prepared;
            assert(missing.size()==31 && missing.front()==1);
            payloads.assign(missing.size(),std::span<const std::byte>(bytes)); return true;
        };
        auto ack=[&](SpillStatus s) { ++saved; if(s==SpillStatus::Present) ++dedup; };
        auto step=[&] { return b.progress(f,prepare,ack); };
        f.probe_busy=true; assert(step()==OwnerSpillBatch::Result::Pending && !b.pending());
        f.probe_busy=false; step(); assert(b.pending() && prepared==0);
        for(unsigned j=0;j<5;++j) { step(); }
        assert(f.probes==2 && prepared==0);
        f.probe->results.fill(SpillStatus::Missing); f.probe->results[0]=SpillStatus::Present;
        f.probe->ready.store(true,std::memory_order_release);
        f.credit=2; step(); assert(prepared==1 && b.batch_submit==2 && saved==1);
        f.writes[0]->status=SpillStatus::Written;
        for(unsigned j=0;j<5;++j) step();
        assert(saved==2 && prepared==1 && b.acknowledged==1); // pending never recounts
        bool refused=false; try { b.reset(); } catch(const std::logic_error&) { refused=true; }
        assert(refused && b.pending()); // cancel cannot reuse borrowed storage
        f.writes[1]->status=SpillStatus::Present; // miss becomes present before write
        f.credit=29; step(); assert(b.batch_submit==31 && prepared==1 && saved==3 && dedup==2);
        assert(f.ids.size()==31 && f.ids.front()==1 && f.ids.back()==31);
        for(auto& t:f.writes) if(DiskKVBridge::poll(t)==SpillStatus::Pending) t->status=SpillStatus::Written;
        assert(step()==OwnerSpillBatch::Result::Complete && saved==32 && !b.pending());
        b.reset(); assert(b.batch_ids.empty() && !b.classified && !b.batch_copy_pending);
        ++scenarios;
    }
    // Cancellation with partial submission drops ONLY unsent pages after ack.
    OwnerSpillBatch b; Fake f; b.batch_ids={{.frontier=1},{.frontier=2}};
    std::byte bytes[4]{}; unsigned copies=0,saved=0;
    auto prep=[&](const auto& m, auto& out){++copies;out.assign(m.size(),std::span<const std::byte>(bytes));return true;};
    auto ack=[&](auto){++saved;};
    b.progress(f,prep,ack); f.probe->results.fill(SpillStatus::Missing); f.probe->ready=true;
    f.credit=1; b.progress(f,prep,ack); assert(b.pending() && b.batch_submit==1);
    f.writes[0]->status=SpillStatus::Written;
    assert(!b.pending()); b.reset(); assert(f.ids.size()==1 && copies==1); ++scenarios;
    // A failed earlier ticket does not allow a later pending borrow to disappear.
    b.batch_ids={{.frontier=3},{.frontier=4}}; b.progress(f,prep,ack);
    f.probe->results.fill(SpillStatus::Missing);f.probe->ready=true;f.credit=2;b.progress(f,prep,ack);
    f.writes[1]->status=SpillStatus::Failed;
    assert(b.progress(f,prep,ack)==OwnerSpillBatch::Result::Failed && b.pending());
    f.writes[2]->status=SpillStatus::Written;assert(!b.pending());b.reset(); ++scenarios;
    std::cout << "owner_spill_batch: PASS scenarios=" << scenarios << " batch=32 families=3 pending/count partial-submit miss-present cancel failure-drain\n";
}
