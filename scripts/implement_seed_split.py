from pathlib import Path
root=Path(__file__).resolve().parents[1]
h=root/'src/targets/qwen3_6/impl/runtime/program.h'
p=root/'src/targets/qwen3_6/impl/runtime/program_impl.h'
s=h.read_text()
s=s.replace('        std::optional<std::uint32_t> l3_restore_frontier;','''        std::optional<std::uint32_t> l3_restore_frontier;
        std::optional<RequestControl::Prefill> staged_prefill;
        struct SeedProgress {
            // Scheduler-only state; accepted CPU reads retain the startup scratch.
            ReadTicket read;
            std::uint32_t phase = 0; // state read, main pages, backend pages, state H2D, commit
            std::uint32_t page = 0;
            bool done = false;
            bool success = false;
        } seed;
        bool sequence_prepared = false;''')
needle='    void start_sequence(std::uint32_t lane, SequenceState& sequence,'
s=s.replace(needle,'''    void prepare_sequence(std::uint32_t lane, SequenceState& sequence,
                          MaterializationTransaction& transaction);
    void finalize_sequence(std::uint32_t lane, SequenceState& sequence,
                           MaterializationTransaction& transaction);
    bool progress_disk_seed(SequenceState& sequence, MaterializationTransaction& transaction);
'''+needle)
h.write_text(s)
s=p.read_text()
a=s.index('        request.prefill.emplace(std::move(prefill));')
b=s.index('        static_assert(std::is_nothrow_move_constructible_v<MaterializationTransaction>);',a)
s=s[:a]+s[a:b].replace('request.prefill','transaction.staged_prefill')+s[b:]
a=s.index('void ProgramImplCore::start_sequence(')
b=s.index('\nruntime::PrefillStepResult\nProgramImplCore::advance_prefill_raw',a)
old=s[a:b]
cut=old.index('        // L3 (disk) tier — read side:')
fin=old.index('        install_sampling(sequence, request, request_plan.sampling);')
prepare=old[:cut].replace('void ProgramImplCore::start_sequence(', 'void ProgramImplCore::prepare_sequence(').replace('!request.prefill','!transaction.staged_prefill').replace('auto& staged                           = *request.prefill;', 'auto& staged                           = *transaction.staged_prefill;')
prepare=prepare.replace('    if (lane >= max_concurrency)', '    if (transaction.sequence_prepared) { throw std::logic_error("sequence prepared twice"); }\n    if (lane >= max_concurrency)',1)
prepare+='''        transaction.sequence_prepared = true;
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
    } catch (...) {
        try { device.synchronize(); } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

void ProgramImplCore::finalize_sequence(std::uint32_t lane, SequenceState& sequence,
                                        MaterializationTransaction& transaction) {
    if (!transaction.sequence_prepared || !transaction.staged_prefill) {
        throw std::logic_error("sequence finalize without prepare");
    }
    auto& request = requests[lane];
    auto& staged = *transaction.staged_prefill;
    auto& request_plan = *transaction.plan->impl_;
    const auto started = Clock::now();
    const auto base = staged.base;
    const auto prompt_tokens = staged.prompt_tokens;
    try {
'''
final=old[fin:]
final=final.replace('        request.lifecycle = Lifecycle::Prefilling;', '''        request.prefill.emplace(std::move(staged));
        transaction.staged_prefill.reset();
        request.lifecycle = Lifecycle::Prefilling;''')
wrapper='''
void ProgramImplCore::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                     MaterializationTransaction& transaction) {
    if (!transaction.sequence_prepared) { prepare_sequence(lane, sequence, transaction); }
    finalize_sequence(lane, sequence, transaction);
}
'''
s=s[:a]+prepare+final+wrapper+s[b:]
s=s.replace('if (!transaction.prepared || !transaction.plan || !destination ||','if ((!transaction.prepared && !transaction.sequence_prepared) || !transaction.plan || !destination ||',1)
s=s.replace('    if (!transaction.prepared) {\n        prepare_materialization(transaction);','    if (!transaction.prepared && !transaction.sequence_prepared) {\n        prepare_materialization(transaction);',1)
needle='    // This is the unique physical publication point.'
pos=s.index(needle)
s=s[:pos]+'''    // Root disk initialization is transaction-owned and never activates a lane.
    // Non-root paths retain their existing atomic start path.
    if (transaction.l3_restore_frontier && transaction.plan->impl_->reuse == ReusePath::Root) {
        auto& sequence = continuation_states[*transaction.root_continuation_index];
        sequence.lane = transaction.destination.value;
        if (!transaction.sequence_prepared) {
            prepare_sequence(sequence.lane, sequence, transaction);
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        if (!progress_disk_seed(sequence, transaction)) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

'''+s[pos:]
needle='    if (cancellation.requested()) { transaction.cancel_pending = true; }\n\n    if (pressure_transition.phase == PressureTransitionPhase::HostReleases)'
s=s.replace(needle,'''    if (cancellation.requested()) { transaction.cancel_pending = true; }
    // Cancellation cannot revoke a worker's borrowed scratch. Poll before ANY
    // abort path (including shutdown/error cleanup) can return the reservation.
    if (transaction.cancel_pending && transaction.seed.read &&
        DiskKVBridge::poll(transaction.seed.read) == DiskReadResult::Pending) {
        out.status = runtime::ContextTransactionStatus::InProgress;
        return out;
    }

    if (pressure_transition.phase == PressureTransitionPhase::HostReleases)''',1)
needle='    const std::uint32_t lane = transaction.destination.value;\n    if (lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Empty)'
s=s.replace(needle,'''    // Exceptional teardown is not the scheduler cancellation path. Keep the
    // borrowed startup allocation alive until the worker's final access.
    while (transaction.seed.read &&
           DiskKVBridge::poll(transaction.seed.read) == DiskReadResult::Pending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    transaction.seed.read.reset();
    transaction.staged_prefill.reset();
    const std::uint32_t lane = transaction.destination.value;
    if (lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Empty)''',1)
p.write_text(s)
