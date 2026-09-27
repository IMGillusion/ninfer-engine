#include "core/seed_lifecycle.h"
#include <cassert>
#include <filesystem>
#include <iostream>
using namespace ninfer;
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::logic_error&) {caught=true;} assert(caught); }
int main() {
    SeedLifecycle once;
    once.begin_prepare(); rejects([&]{once.begin_prepare();});
    once.prepared(); rejects([&]{once.begin_prepare();});
    once.finalize(); rejects([&]{once.finalize();});
    SeedLifecycle early; rejects([&]{early.finalize();});
    auto read=std::make_shared<ReadCompletion>();
    SeedLifecycle cancelled; cancelled.begin_prepare(); cancelled.prepared();
    assert(!cancelled.cancel_and_drained(read));
    rejects([&]{cancelled.abort(read);}); rejects([&]{cancelled.finalize();});
    read->result.store(DiskReadResult::IOFailure,std::memory_order_release);
    assert(cancelled.cancel_and_drained(read)); cancelled.abort(read);
    assert(cancelled.phase()==SeedLifecycle::Phase::Aborted);
    rejects([&]{cancelled.begin_prepare();});
    // Exercise the guard with actual worker/store tickets, not a scheduler loop.
    const auto path=std::filesystem::temp_directory_path()/"ninfer-seed-lifecycle";
    std::filesystem::remove_all(path); std::filesystem::create_directories(path);
    {
        DiskKVBridge bridge({.base_path=path.string(),.main_page_stride=128,
                             .capacity_bytes=1024*1024});
        std::vector<std::byte> bytes(128,std::byte{0x5a}), dst(128);
        const DiskKVIdentity id{.lo=55,.hi=77,.tag=1,.frontier=64};
        assert(bridge.spill_page_sync(id,DiskKVKind::MainKV,bytes));
        auto t=bridge.try_read(id,DiskKVKind::MainKV,dst); assert(t);
        SeedLifecycle cancelled_actual; cancelled_actual.begin_prepare(); cancelled_actual.prepared();
        cancelled_actual.cancel_and_drained(t);
        bridge.wait_idle();
        assert(cancelled_actual.cancel_and_drained(t)); cancelled_actual.abort(t);
        assert(dst==bytes);
        auto missing=bridge.try_read({.lo=99},DiskKVKind::MainKV,dst);assert(missing);
        bridge.wait_idle(); assert(DiskKVBridge::poll(missing)==DiskReadResult::Missing);
        SeedLifecycle fallback; fallback.begin_prepare(); fallback.prepared(); fallback.finalize();
    }
    std::filesystem::remove_all(path);
    std::cout<<"seed lifecycle: one-time prepare/finalize, cancel drain, error and fallback PASS\n";
}
