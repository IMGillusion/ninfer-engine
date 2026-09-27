// Standalone Linux test; all fixtures under a private /tmp directory.
#include "core/disk_kv_store.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace ninfer;
namespace fs = std::filesystem;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error("line " + std::to_string(__LINE__) + ": " #x); } while (0)
static DiskKVIdentity id(unsigned n) { return {n, n+17, n%3, n*64}; }
static std::vector<std::byte> page(unsigned n) { return std::vector<std::byte>(256, std::byte(n)); }
static DiskKVStore::Options options(const std::string& p, unsigned slots=4, bool defer=false) {
    return {.path=p,.slot_size=256,.max_slots=slots,.verify_crc=true,.defer_index_updates=defer};
}
static ino_t inode(const std::string& p) { struct stat s{}; REQUIRE(stat(p.c_str(), &s)==0); return s.st_ino; }
static std::string content(const std::string& p) { std::ifstream f(p,std::ios::binary); return {std::istreambuf_iterator<char>(f),{}}; }
template<class F> static void child(F f) {
    auto p=fork(); REQUIRE(p>=0);
    if (!p) { try { f(); _exit(0); } catch (...) { _exit(2); } }
    int status=0; REQUIRE(waitpid(p,&status,0)==p); REQUIRE(WIFEXITED(status)&&WEXITSTATUS(status)==0);
}
static void evictions(const std::string& p) {
    auto o=options(p); DiskKVStore s(o);
    REQUIRE(s.evict_until_free(0).empty()); REQUIRE(s.evict_until_free(99).empty());
    for(unsigned n=1;n<=4;++n) REQUIRE(s.upsert_page(id(n),page(n)));
    REQUIRE(s.touch(id(1))); auto gone=s.evict_until_free(2);
    REQUIRE(gone.size()==2 && gone[0]==id(2) && gone[1]==id(3));
    REQUIRE(s.live_slots()==2 && s.used_bytes()==8192 && s.free_bytes()==8192);
    REQUIRE(s.evict_until_free(2).empty());
    REQUIRE(s.evict_until_free(99).size()==2); REQUIRE(s.live_slots()==0);
    REQUIRE(s.upsert_page(id(9),page(9))); REQUIRE(s.live_slots()==1);
    DiskKVStore reopened(o); REQUIRE(reopened.live_slots()==1 && reopened.contains(id(9)));
}
static void publication(const std::string& p) {
    auto o=options(p,128,true); DiskKVStore s(o);
    auto before=inode(p+".idx"); auto bytes=content(p+".idx");
    for(unsigned n=0;n<64;++n) REQUIRE(s.upsert_page(id(n),page(n)));
    REQUIRE(inode(p+".idx")==before && content(p+".idx")==bytes);
    s.flush_index(); REQUIRE(inode(p+".idx")!=before);
    before=inode(p+".idx"); s.flush_index(); REQUIRE(inode(p+".idx")==before);
    { DiskKVStore r(o); REQUIRE(r.live_slots()==64); std::vector<std::byte> dst(256);
      for(unsigned n=0;n<64;++n) { REQUIRE(r.read_page(id(n),dst)); REQUIRE(dst==page(n)); } }
    REQUIRE(s.evict(id(0))); { DiskKVStore r(o); REQUIRE(!r.contains(id(0))); }
    REQUIRE(s.evict_until_free(128).size()==63); { DiskKVStore r(o); REQUIRE(r.live_slots()==0); }
}
static void crashes(const std::string& root) {
    auto eager=options(root+"/eager",1);
    child([&] { auto* s=new DiskKVStore(eager); REQUIRE(s->upsert_page(id(1),page(1))); });
    { DiskKVStore s(eager); REQUIRE(s.contains(id(1))); }
    auto deferred=eager; deferred.defer_index_updates=true;
    child([&] { auto* s=new DiskKVStore(deferred); REQUIRE(s->upsert_page(id(2),page(2))); });
    // Persisted row A points at a slot now containing B: neither is advertised.
    { DiskKVStore s(eager); REQUIRE(!s.index_rebuilt_from_scan()); REQUIRE(s.live_slots()==0);
      std::vector<std::byte> dst(256,std::byte{77}); REQUIRE(!s.read_page(id(1),dst)); REQUIRE(dst==page(77));
      REQUIRE(s.upsert_page(id(3),page(3))); }
    { DiskKVStore s(eager); std::vector<std::byte> dst(256); REQUIRE(s.read_page(id(3),dst)); REQUIRE(dst==page(3)); }
    auto scanned=options(root+"/scan",1,true);
    child([&] { auto* s=new DiskKVStore(scanned); REQUIRE(s->upsert_page(id(7),page(7))); });
    fs::remove(scanned.path+".idx");
    { DiskKVStore s(scanned); REQUIRE(s.index_rebuilt_from_scan()); std::vector<std::byte> dst(256);
      REQUIRE(s.read_page(id(7),dst)); REQUIRE(dst==page(7)); }
    // Torn payload under an intact identity must miss without changing dst.
    { std::fstream f(scanned.path,std::ios::in|std::ios::out|std::ios::binary); f.seekp(48+10); f.put(char(42)); }
    { DiskKVStore s(scanned); std::vector<std::byte> dst(256,std::byte{77}); REQUIRE(!s.read_page(id(7),dst)); REQUIRE(dst==page(77)); }
    fs::remove(scanned.path+".idx");
    { DiskKVStore s(scanned); REQUIRE(s.live_slots()==0); }
}
static void retry_flush(const std::string& p) {
    auto o=options(p,4,true); DiskKVStore s(o); auto old=content(p+".idx");
    REQUIRE(s.upsert_page(id(1),page(1))); fs::create_directory(p+".idx.tmp");
    s.flush_index(); REQUIRE(content(p+".idx")==old);
    fs::remove(p+".idx.tmp"); s.flush_index(); REQUIRE(content(p+".idx")!=old);
    DiskKVStore r(o); REQUIRE(r.contains(id(1)));
}
static void concurrent(const std::string& p) {
    DiskKVStore s(options(p,32,true)); std::atomic<bool> done=false; std::atomic<unsigned> reads=0;
    std::thread reader([&] { do { REQUIRE(s.live_slots()<=32); REQUIRE(s.used_bytes()<=32*4096);
        REQUIRE(s.free_bytes()<=32*4096); ++reads; } while(!done.load()); });
    for(unsigned n=0;n<4000;++n) { REQUIRE(s.upsert_page(id(n),page(n))); if(n%31==0) (void)s.evict_until_free(8); }
    done=true; reader.join(); REQUIRE(reads>0); s.flush_index();
    REQUIRE(s.used_bytes()+s.free_bytes()==32*4096);
    std::printf("concurrent snapshots=%u\n",reads.load());
}
static void bench(const std::string& p,bool defer) {
    constexpr unsigned count=2048; auto o=options(p,count,defer); DiskKVStore s(o);
    unsigned rewrites=0; auto start=std::chrono::steady_clock::now();
    for(unsigned n=0;n<count;++n) { auto old=inode(p+".idx"); REQUIRE(s.upsert_page(id(n),page(n)));
      if(defer && (n+1)%64==0) { s.flush_index(); }
      if(inode(p+".idx")!=old) { ++rewrites; }
    }
    s.flush_index(); auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    REQUIRE(rewrites==(defer?count/64:count));
    std::printf("bench deferred=%d pages=%u rewrites=%u elapsed_ms=%.3f\n",defer,count,rewrites,ms);
}
int main() {
    char dir[]="/tmp/dks-audit-XXXXXX"; REQUIRE(mkdtemp(dir)); std::string root=dir;
    try { evictions(root+"/evict"); std::puts("PASS eviction boundaries/LRU/restart");
      publication(root+"/publish"); std::puts("PASS deferred publication/clean flush/eager deletion");
      crashes(root); std::puts("PASS process-exit/stale identity/orphan reuse/scan/CRC");
      retry_flush(root+"/retry"); std::puts("PASS failed publication retry");
      concurrent(root+"/threads"); std::puts("PASS concurrent getters/mutation");
      bench(root+"/bench-eager",false); bench(root+"/bench-batch",true);
      fs::remove_all(root); std::puts("ALL AUDIT TESTS PASSED"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s (fixture %s)\n",e.what(),root.c_str()); return 1; }
}
