from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=r/'src/targets/qwen3_6/impl/state/state_image.cpp'
s=p.read_text(); anchor='void StateImageDevicePool::copy_from_host('
code='''bool StateImageDevicePool::copy_from_host_chunk(HostStateImageConstView source,
    std::int32_t destination, std::size_t& cursor, std::size_t max_bytes, cudaStream_t stream) {
    validate_slot(destination, slot_count(), "StateImage chunk destination out of range");
    validate_host_layout(source.layout, source.data);
    if (max_bytes == 0) { throw std::invalid_argument("empty StateImage copy budget"); }
    std::size_t offset = 0;
    bool submitted = false;
    const auto region = [&](const Tensor& tensor, std::size_t host_offset) {
        const auto end = offset + tensor.bytes();
        if (!submitted && cursor < end) {
            if (cursor < offset) { throw std::logic_error("invalid StateImage cursor"); }
            const auto local = cursor - offset;
            const auto count = std::min(max_bytes, tensor.bytes() - local);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(tensor.data) + local,
                byte_offset(source.data, host_offset + local), count, cudaMemcpyHostToDevice, stream));
            cursor += count;
            submitted = true;
        }
        offset = end;
    };
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        region(linear_.conv_slot(layer,destination), host_layout_.linear_conv.offset +
               layer * host_layout_.linear_conv_layer_bytes);
        region(linear_.recurrent_slot(layer,destination), host_layout_.linear_recurrent.offset +
               layer * host_layout_.linear_recurrent_layer_bytes);
    }
    region(continuation_hidden_slot(destination),host_layout_.continuation_hidden.offset);
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const auto view = dflash_local_->layer_view(layer);
            region(view.k.slice(3,destination,1),host_layout_.dflash_local_k->offset +
                   layer * host_layout_.dflash_local_layer_bytes);
            region(view.v.slice(3,destination,1),host_layout_.dflash_local_v->offset +
                   layer * host_layout_.dflash_local_layer_bytes);
        }
    }
    if (cursor > offset) { throw std::logic_error("StateImage cursor exceeds payload"); }
    return cursor == offset;
}

'''
s=s.replace(anchor,code+anchor,1);p.write_text(s)
p=r/'src/targets/qwen3_6/impl/runtime/program.h';s=p.read_text().replace('            std::uint32_t page = 0;\n            bool done', '            std::uint32_t page = 0;\n            std::size_t state_cursor = 0;\n            bool done');a=s.index('    // probe marked a restorable boundary.'); a=s.rfind('    //',0,a); b=s.index('    [[nodiscard]] PrefillProgress wrap_prefill',a);s=s[:a]+s[b:];p.write_text(s)
p=r/'src/targets/qwen3_6/impl/runtime/program_impl.h';s=p.read_text();s=s.replace('''        // Known remaining bound: whole-image state H2D (not segmented yet).
        state_images->copy_from_host(host_state_images->view(*l3_state_scratch),
                                     state_selectors(sequence).destination, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        ++progress.phase;''','''        const bool complete = state_images->copy_from_host_chunk(
            host_state_images->view(*l3_state_scratch), state_selectors(sequence).destination,
            progress.state_cursor, 4U * 1024U * 1024U, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        if (complete) { ++progress.phase; }''');p.write_text(s)
