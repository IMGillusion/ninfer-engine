#include <bits/stdc++.h>
#define private public
#include "core/disk_kv_bridge.h"
#undef private
#include "core/host_kv_prewrite.h"
#include <cassert>
#include <filesystem>
#include <iostream>
using namespace ninfer;
struct Candidate { unsigned generation=0; };
struct FakeBridge {
 bool busy=true; SpillTicket ticket; std::vector<std::byte> owned;
 SpillTicket try_submit_owned(const DiskKVIdentity&,DiskKVKind,std::vector<std::byte>& b) {
  if(busy)return {};
  owned.swap(b); return ticket=std::make_shared<SpillCompletion>();
 }
};
int main(){
 unsetenv("NINFER_HOST_KV_PREWRITE");
 HostKVPrewrite<Candidate,2> q; assert(!q.enabled); assert(!q.add({1}));q.enabled=true;
 FakeBridge fake; unsigned generation=1,copies=0;
 auto valid=[&](auto c){return c.generation==generation;};
 auto copy=[&](auto,auto& id,auto& kind,auto& bytes){++copies;id={.lo=42,.frontier=256};kind=DiskKVKind::MainKV;bytes.assign(64,std::byte{42});return true;};
 assert(q.add({1}));assert(q.add({1}));assert(!q.add({1}));assert(q.stats.full==1);
 q.tick(fake,valid,copy); assert(q.pending()==2&&q.stats.busy==1&&copies==1&&q.buffered()==64);
 q.tick(fake,valid,copy); assert(copies==1&&q.stats.written==0);
 generation=2;q.tick(fake,valid,copy);assert(q.pending()==0&&q.stats.invalid==2&&q.buffered()==0);
 assert(q.add({2}));fake.busy=false;q.tick(fake,valid,copy);assert(q.pending()==0&&q.buffered()==64);
 generation=3;assert(fake.owned[0]==std::byte{42});q.tick(fake,valid,copy);assert(q.stats.written==0);
 fake.ticket->status.store(SpillStatus::Written);q.tick(fake,valid,copy);assert(q.stats.written==1&&q.buffered()==0);
 assert(q.add({3}));q.tick(fake,valid,copy);fake.ticket->status.store(SpillStatus::Failed);q.tick(fake,valid,copy);assert(q.stats.failed==1);
 // A slow single memcpy-equivalent stops admission at the quantum boundary.
 HostKVPrewrite<Candidate,2> timed;timed.enabled=true;assert(timed.add({3}));
 auto slow=[&](auto c,auto& id,auto& kind,auto& bytes){
  const bool result=copy(c,id,kind,bytes);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));return result;
 };
 timed.tick(fake,valid,slow);assert(timed.pending()==1&&timed.buffered()==64);
 timed.tick(fake,valid,copy);assert(timed.pending()==0);
 fake.ticket->status.store(SpillStatus::Written);timed.tick(fake,valid,copy);
 assert(timed.stats.written==1);
 HostKVPrewrite<Candidate,2> throwing;throwing.enabled=true;assert(throwing.add({3}));
 throwing.tick(fake,valid,[](auto,auto&,auto&,auto&)->bool{throw std::bad_alloc();});
 assert(throwing.stats.failed==1&&throwing.pending()==0);
 const auto root=std::filesystem::temp_directory_path()/"host-prewrite-test";std::filesystem::remove_all(root);std::filesystem::create_directories(root);
 {
 DiskKVBridge bridge({.base_path=root.string(),.main_page_stride=64,.backend_page_stride=64,.capacity_bytes=4*1024*1024});
 for(auto kind:{DiskKVKind::MainKV,DiskKVKind::BackendKV}) {
  std::vector<std::byte> bytes(64,std::byte{77});const DiskKVIdentity id{.lo=123,.frontier=256};SpillTicket ticket;
  while(!(ticket=bridge.try_submit_owned(id,kind,bytes))) std::this_thread::yield();
  assert(bytes.empty());bytes.assign(64,std::byte{99});bridge.wait_idle();assert(DiskKVBridge::poll(ticket)==SpillStatus::Written);
  std::array<std::byte,64> out{};assert(bridge.restore_page(id,kind,out));assert(out[0]==std::byte{77});
 }
 assert(bridge.stats().queue_drops==0);
 // Deterministic real queue saturation: workers block on store, never engine.
 {
  std::lock_guard hold(bridge.families_[0].store->mu_);
  for(unsigned i=0;i<512;++i) {
   SpillTicket t;
   while(!(t=bridge.try_probe({.lo=999},DiskKVKind::MainKV))) std::this_thread::yield();
  }
  std::vector<std::byte> retry(64,std::byte{88});
  assert(!bridge.try_submit_owned({.lo=1000},DiskKVKind::MainKV,retry));
  assert(retry.size()==64 && retry[0]==std::byte{88});
  assert(bridge.stats().queue_drops==0);
 }
 bridge.wait_idle();
 std::vector<std::byte> retry(64,std::byte{88});SpillTicket retried;
 while(!(retried=bridge.try_submit_owned({.lo=1000},DiskKVKind::MainKV,retry))) std::this_thread::yield();
 bridge.wait_idle();assert(DiskKVBridge::poll(retried)==SpillStatus::Written);
 std::array<std::byte,64> out{};assert(bridge.restore_page({.lo=1000},DiskKVKind::MainKV,out));assert(out[0]==std::byte{88});
 }
 std::filesystem::remove_all(root);
 std::cout<<"PASS default-off full invalid-generation busy-retry no-recopy owned-readback two-families failed-ticket drops=0\n";
}
