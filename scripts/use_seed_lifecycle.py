from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=r/'src/targets/qwen3_6/impl/runtime/program.h';s=p.read_text().replace('#pragma once','#pragma once\n#include "core/seed_lifecycle.h"',1).replace('        bool sequence_prepared = false;', '        SeedLifecycle seed_lifecycle;');p.write_text(s)
p=r/'src/targets/qwen3_6/impl/runtime/program_impl.h';s=p.read_text().replace('transaction.sequence_prepared','transaction.seed_lifecycle.is_prepared()');s=s.replace('    if (transaction.seed_lifecycle.is_prepared()) { throw std::logic_error("sequence prepared twice"); }','    transaction.seed_lifecycle.begin_prepare();');s=s.replace('        transaction.seed_lifecycle.is_prepared() = true;', '        transaction.seed_lifecycle.prepared();');s=s.replace('''    if (transaction.cancel_pending && transaction.seed.read &&
        DiskKVBridge::poll(transaction.seed.read) == DiskReadResult::Pending) {''','''    if (transaction.cancel_pending &&
        !transaction.seed_lifecycle.cancel_and_drained(transaction.seed.read)) {''');s=s.replace('''    transaction.seed.read.reset();
    transaction.staged_prefill.reset();''','''    transaction.seed_lifecycle.abort(transaction.seed.read);
    transaction.seed.read.reset();
    transaction.staged_prefill.reset();''',1);s=s.replace('''        transaction.staged_prefill.reset();
        request.lifecycle = Lifecycle::Prefilling;''','''        transaction.staged_prefill.reset();
        request.lifecycle = Lifecycle::Prefilling;''')
# Publication guard is validated at finalize entry, before any move/copy. It must
# stay Prepared until start_request has verified entitlement; no additional throws
# after terminal transition. Cleanup after publication failure remains legal.
s=s.replace('''        invalidate_lane(lane);
        const SequenceHandle handle =''','''        transaction.seed_lifecycle.finalize();
        invalidate_lane(lane);
        const SequenceHandle handle =''',1)
p.write_text(s)
