// Included inside the variant runtime namespace by program_impl.h.
bool ProgramImplCore::proactive_pressure() noexcept {
    if (!proactive_enabled_ || !disk_kv || !host_kv_arena || !state_images) return false;
    const auto used = physical_occupancy();
    const auto capacity = admission_capacity();
    const std::array<std::pair<std::uint64_t,std::uint64_t>,4> usage{{
        {host_kv_arena->occupied_bytes(),host_kv_arena->capacity_bytes()},
        {host_state_images->occupied(),host_state_images->capacity()},
        {used.device.main_kv_pages,capacity.device.main_kv_pages},
        {used.device.state_slots,capacity.device.state_slots}}};
    const bool active = proactive_watermarks_.update(usage);
    // Name the trigger decision: without this the feature is blind in production
    // (a silent "no release" cannot be told apart from "never triggered").
    const auto now = std::chrono::steady_clock::now();
    if (active != proactive_log_pressure_ || now - proactive_last_log_ > std::chrono::seconds(30)) {
        proactive_log_pressure_ = active;
        proactive_last_log_ = now;
        const auto pct = [](std::uint64_t u, std::uint64_t c) {
            return c ? static_cast<unsigned>(static_cast<long double>(u) * 100.0L / c) : 0U;
        };
        disk_kv_logf('I', "l3-proactive",
                     "pressure | active=%d host_kv=%u%% device_kv=%u%% host_state=%u%% device_state=%u%%",
                     static_cast<int>(active), pct(usage[0].first, usage[0].second),
                     pct(usage[2].first, usage[2].second), pct(usage[1].first, usage[1].second),
                     pct(usage[3].first, usage[3].second));
    }
    return active;
}

bool ProgramImplCore::proactive_begin(const ContinuationHandle& owner) {
    // Every refusal names itself: a silent "no release ever" is indistinguishable
    // from "never triggered" in production, and the conditions are many.
    const auto fail=[&](const char* why) { proactive_begin_fail_=why; return false; };
    proactive_begin_fail_=nullptr;
    if (proactive_) return fail("busy");
    if (!proactive_pressure()) return fail("no-pressure");
    if (!disk_kv) return fail("no-disk-tier");
    // The shared seed/KV scratch is single-user: a live context transaction may
    // hold borrowed scratch bytes with tickets still in flight.
    if (has_context_transaction() || pending_transaction_) return fail("transaction");
    if (!valid_continuation(owner)) return fail("invalid-owner");
    if (has_unsettled_state_fork()) return fail("state-fork");
    const auto index=ContractAccess::index(owner);
    auto& sequence=continuation_states[index];
    if (!can_release_continuation_slot_strict(index)) return fail("not-releasable");
    if (!sequence.shared_prefix_references.empty()) return fail("shares-prefix");
    auto work=std::make_unique<ProactiveWork>();
    work->index=index;
    work->generation=ContractAccess::epoch(owner);
    // Victim-local fingerprint: a GLOBAL resource revision changes on every
    // unrelated release/capture, which cancelled every preparation before it
    // could finish. What matters is whether THIS owner's persisted frontier moved.
    work->frontier=sequence.execution_frontier;
    work->ledger=sequence.ledger_frontier;
    // Drive the engine's OWN resumable owner spill (the same batched machinery the
    // on-demand materialization uses). A hand-rolled per-page closure was ~100x
    // slower: every page paid two store-queue round trips plus a read-back.
    work->spill.id = index;
    work->spill.endpoint_valid_at_start = sequence.endpoint_valid;
    work->spill.seam_valid_at_start =
        sequence.rewrite_checkpoint.valid && static_cast<bool>(sequence.rewrite_state);
    proactive_=std::move(work);
    return true;
}

bool ProgramImplCore::proactive_begin_checkpoint(const ContinuationHandle& owner) {
    // Every refusal names itself (same discipline as proactive_begin): a silent
    // "never checkpointed" is indistinguishable from "restore is broken".
    const auto fail=[&](const char* why) { proactive_begin_fail_=why; return false; };
    proactive_begin_fail_=nullptr;
    if (!checkpoint_enabled_) return fail("checkpoint-disabled");
    if (proactive_) return fail("busy");
    if (!disk_kv) return fail("no-disk-tier");
    // The shared seed/KV scratch is single-user: a live context transaction may
    // hold borrowed scratch bytes with tickets still in flight.
    if (has_context_transaction() || pending_transaction_) return fail("transaction");
    if (!valid_continuation(owner)) return fail("invalid-owner");
    if (has_unsettled_state_fork()) return fail("state-fork");
    const auto index=ContractAccess::index(owner);
    auto& sequence=continuation_states[index];
    // No watermark here (a checkpoint is exactly the zero-pressure path) and no
    // strict-releasable gate (nothing gets released). But a checkpoint exists to
    // serve a later restore: without a valid endpoint state the spill cannot
    // capture stage-2 and the persisted chain would be unrecoverable.
    if (!sequence.endpoint_valid) return fail("endpoint-invalid");
    auto work=std::make_unique<ProactiveWork>();
    work->checkpoint_only=true;
    work->index=index;
    work->generation=ContractAccess::epoch(owner);
    work->frontier=sequence.execution_frontier;
    work->ledger=sequence.ledger_frontier;
    // Same resumable owner-spill machinery as the eviction path; the only
    // difference is what commit does with the finished spill.
    work->spill.id = index;
    work->spill.endpoint_valid_at_start = sequence.endpoint_valid;
    work->spill.seam_valid_at_start =
        sequence.rewrite_checkpoint.valid && static_cast<bool>(sequence.rewrite_state);
    proactive_=std::move(work);
    return true;
}

void ProgramImplCore::proactive_cancel() noexcept {
    if (!proactive_) return;
    cleanup_owner_spill(proactive_->spill);
    proactive_.reset();
}

int ProgramImplCore::proactive_tick(const ContinuationHandle& owner) noexcept {
    if (!proactive_) return -1;
    auto& work=*proactive_;
    if (!valid_continuation(owner) || ContractAccess::index(owner)!=work.index ||
        ContractAccess::epoch(owner)!=work.generation ||
        continuation_states[work.index].execution_frontier!=work.frontier ||
        continuation_states[work.index].ledger_frontier!=work.ledger) {
        proactive_begin_fail_="tick-stale"; proactive_cancel(); return -1;
    }
    // PAUSE (do not discard) while a context transaction is live: it may hold
    // borrowed seed/KV scratch bytes with tickets in flight, and the spill path
    // uses that same single-user scratch. The cursor and the already written chain
    // survive; the next tick resumes after re-validating this fingerprint.
    if (has_context_transaction() || pending_transaction_) {
        // Layer 1 of the 500-race fix (mutual exclusion): an owner that an open
        // materialization transaction needs as its source or as a victim must
        // not keep its pages pinned by a paused in-flight proactive spill —
        // those pins used to span the transaction's destructive source
        // truncation ("consumed source KV is not destructively truncatable").
        // Drop the pinned batch and back off instead: the persisted chain is
        // content-addressed, so a later re-spill dedups down to the increment.
        if (auto* transaction =
                std::get_if<MaterializationTransaction>(&context_transaction_)) {
            const bool blocked =
                (transaction->has_source && transaction->source_index == work.index) ||
                std::find(transaction->victim_indices.begin(),
                          transaction->victim_indices.end(),
                          work.index) != transaction->victim_indices.end();
            if (blocked) {
                proactive_begin_fail_ = "owner-in-open-transaction";
                proactive_cancel();
                return -1;
            }
        }
        return 0;
    }
    try {
        if (progress_owner_spill(continuation_states[work.index], work.spill)) { return 1; }
        if (work.spill.failed) { proactive_begin_fail_="spill-failed"; proactive_cancel(); return -1; }
        return 0;
    } catch (...) { proactive_begin_fail_="tick-exception"; proactive_cancel(); return -1; }
}

bool ProgramImplCore::proactive_commit(ContinuationHandle& owner) noexcept {
    if (!proactive_ || proactive_->spill.failed) return false;
    if (!valid_continuation(owner) || ContractAccess::index(owner)!=proactive_->index ||
        ContractAccess::epoch(owner)!=proactive_->generation ||
        continuation_states[proactive_->index].execution_frontier!=proactive_->frontier ||
        continuation_states[proactive_->index].ledger_frontier!=proactive_->ledger) return false;
    if (proactive_->checkpoint_only) {
        // Checkpoint mode: the persisted spill IS the product. The owner stays
        // catalogued and fully resident — no release, no consume, no resource
        // revision bump (the chain on disk is content-addressed, so re-spilling
        // after later turns dedups against these pages: the increment only).
        try {
            const auto frontier=proactive_->frontier;
            const auto saved=proactive_->spill.saved;
            const auto dedup=proactive_->spill.dedup;
            const auto state=proactive_->spill.state_saved;
            cleanup_owner_spill(proactive_->spill);
            proactive_.reset();
            disk_kv_logf('I',"l3-proactive","checkpoint | frontier=%u saved=%llu dedup=%llu state=%llu",
                frontier,(unsigned long long)saved,(unsigned long long)dedup,
                (unsigned long long)state);
            return true;
        } catch (...) { proactive_cancel(); return false; }
    }
    try {
        if (!can_release_continuation_slot_strict(proactive_->index)) return false;
        const auto& spill=proactive_->spill;
        const auto before=physical_occupancy();
        cleanup_owner_spill(proactive_->spill);
        release_continuation_slot_strict(proactive_->index, true);
        ContractAccess::consume(owner);
        // The logical release only drops REFERENCES. Without this the owner's host
        // KV extents stay allocated and the release returns almost nothing (measured:
        // host_kv_bytes=0 on every release while the session's KV is host-resident).
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        advance_resource_revision();
        const auto after=physical_occupancy();
        const auto host_bytes=before.host.kv_bytes-after.host.kv_bytes;
        proactive_released_host_bytes_+=host_bytes;
        disk_kv_logf('I',"l3-proactive",
                     "released | host_kv_bytes=%llu device_kv_pages=%u host_state_slots=%u device_state_slots=%u | verified_disk_bytes=%llu saved=%llu dedup=%llu d2h=%llu state=%llu",
            (unsigned long long)host_bytes,before.device.main_kv_pages-after.device.main_kv_pages,
            before.host.state_slots-after.host.state_slots,before.device.state_slots-after.device.state_slots,
            (unsigned long long)(spill.saved*text_host_kv_page_stride),
            (unsigned long long)spill.saved,(unsigned long long)spill.dedup,(unsigned long long)spill.d2h,
            (unsigned long long)spill.state_saved);
        proactive_.reset(); return true;
    } catch (...) { proactive_cancel(); return false; }
}
