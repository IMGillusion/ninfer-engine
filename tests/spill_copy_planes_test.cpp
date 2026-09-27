// Link-time CUDA backend injection: execute the REAL pool D2H plane loop.
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#define private public
#include "core/host_kv_arena.h"
#undef private
#include <sys/wait.h>
#include <unistd.h>
using namespace ninfer;
namespace ninfer {
bool HostKVAllocationView::valid() const noexcept { return data_ && layout_; }
const HostKVPageLayout& HostKVAllocationView::layout() const { return *layout_; }
}
static int fail_at, calls, sync_error, drains;
static bool pinned=true, disk=false;
static std::vector<std::function<void()>> pending;
extern "C" cudaError_t cudaMemcpy2DAsync(void* d,size_t dp,const void* s,size_t sp,size_t w,size_t h,cudaMemcpyKind,cudaStream_t) {
 assert(pinned&&!disk); if(++calls==fail_at) return cudaErrorInvalidValue;
 pending.emplace_back([=]{for(size_t i=0;i<h;++i)memcpy((char*)d+i*dp,(char*)s+i*sp,w);});return cudaSuccess;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) {
 assert(pinned&&!disk);++drains;if(sync_error)return cudaErrorLaunchFailure;
 for(auto& f:pending)f();pending.clear();return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
void run(PagedKVPlaneOrder order,int fail,int sync) {
 fail_at=fail;calls=drains=0;sync_error=sync;pending.clear();pinned=true;disk=false;
 KVPageGeometry geometry{.device_plane_order=order,.planes={{DType::I8,8,2,256},{DType::I8,8,2,256},{DType::I8,8,2,256}}};
 LayoutBuilder builder;auto layout=plan_device_kv_page_pool(builder,{.page_group_count=4,.geometry=geometry});
 const auto size=builder.finish(256);
 std::unique_ptr<void,decltype(&std::free)> storage(std::aligned_alloc(256,size),&std::free);
 DeviceKVPagePool pool({static_cast<std::byte*>(storage.get()),size},layout);
 auto reservation=pool.reserve(4);std::vector<DeviceKVPageLease> leases;leases.reserve(4);pool.materialize(*reservation,4,leases);
 HostKVPageLayout hl; hl.geometry=geometry;
 for(const auto& spec:geometry.planes){ (void)spec; hl.planes.push_back({hl.page_stride,1024,512});hl.page_stride+=1024;}std::vector<std::byte> bytes(hl.page_stride*3);
 // Non-owning valid view: no host allocator CUDA allocation needed in CPU test.
 HostKVAllocationView view({reinterpret_cast<const HostKVArena*>(1),0,0},bytes.data(),&hl,3);
 for(size_t plane=0;plane<pool.plane_count();++plane){auto& t=pool.plane(plane);for(size_t i=0;i<t.bytes();++i)((unsigned char*)t.data)[i]=(i*17+plane*51+i/256)%251;}
 std::array<DeviceKVPageHandle,3> handles{leases[2].handle(),leases[0].handle(),leases[1].handle()};
 bool ok=pool.spill_copy_to_host(handles,view,nullptr);
 assert(drains==1&&pending.empty());pinned=false;
 if(fail){assert(!ok&&calls==fail&&!disk);return;}
 assert(ok);disk=true;
 for(size_t p=0;p<3;++p)for(size_t n=0;n<pool.plane_count();++n){
  auto& t=pool.plane(n);auto& hp=hl.planes[n];int phys=handles[p].index_;
  auto* out=bytes.data()+p*hl.page_stride+hp.offset;
  if(order==PagedKVPlaneOrder::PageMajor)assert(!memcmp(out,(char*)t.data+phys*t.nb[3],hp.page_payload_bytes));
  else for(int head=0;head<t.ne[3];++head)assert(!memcmp(out+head*hp.head_payload_bytes,(char*)t.data+head*t.nb[3]+phys*t.nb[2],hp.head_payload_bytes));
 }
}
int main(){
 for(auto order:{PagedKVPlaneOrder::PageMajor,PagedKVPlaneOrder::HeadMajor}){
  run(order,0,0);int total=calls;
  for(int fail:{1,total/2,total})run(order,fail,0);
  auto pid=fork();assert(pid>=0);if(!pid){run(order,3,1);_exit(99);}int s;waitpid(pid,&s,0);assert(WIFEXITED(s)&&WEXITSTATUS(s)==70);
 }
 puts("spill_copy_planes PASS: real KV loops both layouts, discontiguous pages, first/middle/last errors, fatal drain");
}
