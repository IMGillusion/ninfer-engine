#include "targets/qwen3_6/impl/runtime/owner_spill_plan.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include <cassert>
#include <iostream>
#include <vector>
using namespace ninfer::targets::qwen3_6;
using namespace ninfer::targets::qwen3_6::detail;
using namespace ninfer::targets::qwen3_6::runtime_support;
static CheckpointSummary checkpoint(unsigned main, unsigned backend) {
    CheckpointSummary c;
    c.required_kv = {main, backend, (main+255)/256, (backend+255)/256};
    return c;
}
int main() {
    ContinuationSummary s;
    s.endpoint = checkpoint(100001,100000);
    s.rewrite = checkpoint(97001,97000);
    s.long_anchors.push_back(checkpoint(96000,95999));
    assert(owner_spill_requirement(s) == s.endpoint->required_kv);
    s.endpoint.reset(); // publish_checkpoint_drop retains historical E, not its KV
    auto r = owner_spill_requirement(s);
    assert(r == s.rewrite->required_kv);
    assert(r.main_pages == 379 && r.backend_pages == 379);
    assert(!owner_spill_page_mapped(379,r.main_pages));
    // Final retained partial page is valid; no access to historical page 379.
    assert(owner_spill_required_columns(r.main_frontier,378,256) == 233);
    assert(owner_spill_page_committed(233,233));
    assert(!owner_spill_page_committed(233,232));
    assert(!owner_spill_page_mapped(378,378)); // required missing page is not clamped
    assert(r.backend_frontier == 97000); // already MTP-adjusted, never subtract again
    s.rewrite.reset();
    assert(owner_spill_requirement(s) == s.long_anchors[0].required_kv);
    s.long_anchors.push_back(checkpoint(98000,97999));
    assert(owner_spill_requirement(s) == s.long_anchors.back().required_kv);
    s.long_anchors.clear();
    assert(owner_spill_requirement(s) == TargetKVRequirement{});
    s.endpoint = checkpoint(512,0); // non-speculative exact page boundary
    assert(owner_spill_requirement(s).backend_frontier == 0);
    assert(owner_spill_required_columns(512,1,256) == 256);
    s.endpoint = checkpoint(513,512); // MTP main tail, backend full page
    r = owner_spill_requirement(s);
    assert(r.main_pages == 3 && r.backend_pages == 2);
    assert(owner_spill_required_columns(513,2,256) == 1);
    PrefixShortlistDigests digests;
    digests.assign(PreparedPromptData{});
    std::vector<ninfer::TokenId> tokens(513,42);
    digests.append_generated(tokens,0);
    assert(digests.size() == 513);
    assert(owner_spill_digest_available(513,digests.size()));
    (void)digests.at(513); // actual type validates inclusive final frontier
    assert(!owner_spill_digest_available(514,digests.size()));
    digests.truncate(512);
    assert(owner_spill_digest_available(512,digests.size()));
    (void)digests.at(512);
    std::cout << "PASS owner_spill_plan: endpoint-drop rewrite/anchors, MTP, boundary/tail, missing/short-commit, real digest boundary\n";
}
