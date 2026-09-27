#include "ninfer/targets/qwen3_6/state_image.h"
#include "core/device.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
int main() {
    StateImageSpec spec;
    spec.linear = {.layers=3,.conv_channels=10,.conv_width=3,.value_heads=4,
                   .value_head_dim=5,.key_head_dim=6,.slot_count=2,.conv_dtype=DType::BF16};
    spec.hidden=23;
    LayoutBuilder builder;
    auto layout=plan_state_image_device_pool(builder,spec);
    DeviceArena arena(builder.finish(256));
    StateImageDevicePool pool({arena.base(),arena.capacity()},layout);
    HostStatePool hosts(layout.host,2);
    auto a=hosts.allocate(), b=hosts.allocate(); assert(a && b);
    auto input=hosts.writable_view(*a), output=hosts.writable_view(*b);
    for(std::size_t i=0;i<layout.host.image_bytes;++i) input.data[i]=std::byte((i*17+3)%251);
    std::memset(output.data,0,layout.host.image_bytes);
    cudaStream_t stream{}; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    pool.zero_all(stream);
    std::size_t cursor=0, calls=0;
    bool complete=false;
    while(!complete) {
        auto prior=cursor;
        complete=pool.copy_from_host_chunk(hosts.view(*a),1,cursor,37,stream);
        assert(cursor>prior && cursor-prior<=37);
        CUDA_CHECK(cudaStreamSynchronize(stream)); ++calls;
    }
    auto end=cursor;
    assert(pool.copy_from_host_chunk(hosts.view(*a),1,cursor,37,stream) && cursor==end);
    pool.copy_to_host(1,output,stream); CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto match=[&](const LayoutRegion& region) {
        assert(std::memcmp(input.data+region.offset,output.data+region.offset,region.bytes)==0);
    };
    match(layout.host.linear_conv); match(layout.host.linear_recurrent);match(layout.host.continuation_hidden);
    bool rejected=false; try {pool.copy_from_host_chunk(hosts.view(*a),1,cursor,0,stream);}
    catch(const std::invalid_argument&){rejected=true;} assert(rejected);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout<<"state H2D real GPU: "<<calls<<" chunks, "<<cursor<<" payload bytes PASS\n";
}
