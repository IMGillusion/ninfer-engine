#include "core/disk_kv_store.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
using namespace ninfer;
static double ms_since(auto& t){ auto n=std::chrono::steady_clock::now(); return std::chrono::duration<double,std::milli>(n-t).count(); }
int main(int argc, char** argv){
    const std::size_t stride=1589632;
    const int pages = argc>1 ? atoi(argv[1]) : 200;
    auto T0=std::chrono::steady_clock::now();
    std::printf("opening (mmap of %d pages x 1.51MiB = %.2fGB)...\n", pages, (double)pages*stride/1e9);
    DiskKVStore s(DiskKVStore::Options{.path="/tmp/diag.bin",.page_stride=stride,.max_pages=(std::uint32_t)(pages+10)});
    std::printf("  open took %.0f ms\n", ms_since(T0));
    auto T1=std::chrono::steady_clock::now();
    auto g=s.reserve(pages);
    std::printf("  reserve %d took %.1f ms, id=%d\n", pages, ms_since(T1), (int)(g?*g:-1));
    std::vector<std::byte> buf(pages*stride, std::byte{0x5A});
    auto T2=std::chrono::steady_clock::now();
    bool ok=s.write_group(*g, std::span<const std::byte>(buf.data(),buf.size()));
    std::printf("  write %d (~%.2fGB) took %.1f ms ok=%d\n", pages,(double)pages*stride/1e9, ms_since(T2), (int)ok);
    auto T3=std::chrono::steady_clock::now();
    bool pub=s.publish(*g);
    std::printf("  publish (crc32) took %.1f ms pub=%d\n", ms_since(T3), (int)pub);
    std::vector<std::byte> out(pages*stride);
    auto T4=std::chrono::steady_clock::now();
    bool rd=s.read_group(*g, std::span<std::byte>(out.data(),out.size()));
    std::printf("  read back took %.1f ms ok=%d match=%d\n", ms_since(T4), (int)rd, (int)(std::memcmp(buf.data(),out.data(),buf.size())==0));
    std::printf("DONE\n");
    return 0;
}
