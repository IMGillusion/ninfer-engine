#include "core/owner_spill_batch.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <thread>
using namespace ninfer;
int main() {
    const auto root = std::filesystem::temp_directory_path()/"owner-spill-r7-test";
    std::filesystem::remove_all(root); std::filesystem::create_directories(root);
    unsigned batches=0;
    {
        DiskKVBridge bridge({.base_path=root.string(),.main_page_stride=64,.backend_page_stride=64,
                             .state_page_stride=64,.capacity_bytes=4*1024*1024});
        std::array<std::array<std::byte,64>,32> bytes{};
        for(unsigned i=0;i<32;++i) for(unsigned j=0;j<64;++j) bytes[i][j]=std::byte((i*7+j)%256);
        for(auto kind : {DiskKVKind::MainKV,DiskKVKind::BackendKV,DiskKVKind::StateImage}) {
            OwnerSpillBatch b; b.batch_kind=kind;
            for(unsigned i=0;i<32;++i)b.batch_ids.push_back({.lo=i+1,.frontier=(i+1)*256});
            assert(bridge.spill_page_sync(b.batch_ids[0],kind,bytes[0]));
            unsigned saved=0,dups=0,prepared=0;
            auto prepare=[&](const auto& missing,auto& out){
                ++prepared; assert(missing.size()==31);
                // Real bridge race: probe Missing -> another writer publishes -> Present ack.
                assert(bridge.spill_page_sync(b.batch_ids[missing[0]],kind,bytes[missing[0]]));
                out.clear(); for(auto i:missing) out.emplace_back(bytes[i]); return true;
            };
            auto ack=[&](SpillStatus s){++saved;if(s==SpillStatus::Present)++dups;};
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            for(;;) {
                auto result=b.progress(bridge,prepare,ack);
                assert(result!=OwnerSpillBatch::Result::Failed);
                if(result==OwnerSpillBatch::Result::Complete)break;
                assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();
            }
            assert(saved==32 && dups==1 && prepared==1 && b.batch_submit==31);
            for(const auto& id:b.batch_ids){std::array<std::byte,64> read{};assert(bridge.restore_page(id,kind,read));assert(read==bytes[id.lo-1]);}
            b.reset(); ++batches;
        }
        bridge.wait_idle();assert(bridge.stats().queue_drops==0);
    }
    std::filesystem::remove_all(root);
    std::cout<<"owner_spill_batch_bridge: PASS batches="<<batches<<" pages=96 readback=96 miss-present=3 drops=0\n";
}
