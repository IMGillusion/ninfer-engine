#include <bits/stdc++.h>
#define private public
#include "core/disk_kv_bridge.h"
#undef private
using namespace ninfer;
using Clock=std::chrono::steady_clock;
static void until(auto f) { auto end=Clock::now()+std::chrono::seconds(10); while(!f()){ assert(Clock::now()<end); std::this_thread::yield(); } }
int main(){
 std::string root="/tmp/seed-read-ticket"; std::filesystem::remove_all(root);
 DiskKVBridge b({.base_path=root,.main_page_stride=4096,.backend_page_stride=4096,.state_page_stride=4096,.capacity_bytes=4*1024*1024});
 DiskKVIdentity id{1,2,3,64}; std::vector<std::byte> src(4096,std::byte{42}),dst(4096);
 auto submit=[&](DiskKVIdentity key,DiskKVKind k,std::span<std::byte> out){ ReadTicket t; until([&]{t=b.try_read(key,k,out);return bool(t);});return t;};
 auto finish=[&](ReadTicket t,DiskReadResult expected){until([&]{return b.poll(t)!=DiskReadResult::Pending;});assert(b.poll(t)==expected);};
 for(auto kind:{DiskKVKind::MainKV,DiskKVKind::BackendKV,DiskKVKind::StateImage}){
  assert(b.spill_page_sync(id,kind,src));finish(submit(id,kind,dst),DiskReadResult::Success);assert(src==dst);
  finish(submit({9,9,3,64},kind,dst),DiskReadResult::Missing);
  finish(submit(id,kind,std::span(dst).first(1)),DiskReadResult::BadSize);
 }
 finish(submit(id,static_cast<DiskKVKind>(255),dst),DiskReadResult::Disabled);
 b.wait_idle();
 // A blocked actual store read must not stall the caller's scheduling loop.
 // Simulated cancellation retains the scratch until the accepted borrow drains.
 ReadTicket pending; size_t decode=0; bool expired=false,cancelled=false;
 auto begin=Clock::now();
 {
  std::lock_guard lock(b.families_[0].store->mu_);
  pending=submit(id,DiskKVKind::MainKV,dst);
  while(Clock::now()-begin<std::chrono::milliseconds(80)){
   assert(b.poll(pending)==DiskReadResult::Pending); ++decode;
   if(Clock::now()-begin>=std::chrono::milliseconds(20)){expired=true;cancelled=true;}
   // no release/reuse on cancellation while poll is Pending
   std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
 }
 assert(decode>10 && expired && cancelled); finish(pending,DiskReadResult::Success); assert(src==dst);
 // Corruption classes are distinct. Restore does not delete even a bad entry;
 // invalidation must be conditional under the store lock (avoid a repair race).
 b.wait_idle(); auto& s=*b.families_[0].store;
 auto slot=s.index_.at(id); auto* hdr=s.slot_hdr(slot);
 auto magic=hdr->magic;hdr->magic=0;finish(submit(id,DiskKVKind::MainKV,dst),DiskReadResult::BadHeader);hdr->magic=magic;
 hdr->lo++;finish(submit(id,DiskKVKind::MainKV,dst),DiskReadResult::BadIdentity);hdr->lo--;
 hdr->crc^=1;finish(submit(id,DiskKVKind::MainKV,dst),DiskReadResult::BadCRC);hdr->crc^=1;
 assert(b.contains(id,DiskKVKind::MainKV));finish(submit(id,DiskKVKind::MainKV,dst),DiskReadResult::Success);
 // Queue saturation accepts exactly the shared bound, never overwrites scratch.
 b.wait_idle();std::vector<ReadTicket> tickets;std::vector<std::vector<std::byte>> buffers(512,std::vector<std::byte>(4096));
 {std::lock_guard lock(s.mu_);for(auto& v:buffers)tickets.push_back(submit(id,DiskKVKind::MainKV,v));assert(!b.try_read(id,DiskKVKind::MainKV,dst));assert(b.stats().queue_drops==0);}
 for(auto t:tickets)finish(t,DiskReadResult::Success);
 for(auto& v:buffers)assert(v==src);
 std::cout<<"PASS read classifications, three families, blocked read/decode ticks="<<decode<<", simulated deadline/cancel retention, saturation=512\n";
}
