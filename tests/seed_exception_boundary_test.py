"""Compile the actual production fail lambda with deterministic borrowed-I/O mocks."""
from pathlib import Path
import subprocess
s=Path('/src/src/targets/qwen3_6/impl/runtime/program_impl.h').read_text()
a=s.index('    const auto fail = [&]() {',s.index('bool ProgramImplCore::progress_disk_seed'))
b=s.index('    if (progress.failing)',a)
body=s[a:b]
prefix=r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include "core/seed_test_hooks.h"
struct Ticket { bool pending=true; };
enum class DiskReadResult { Pending, Success };
struct DiskKVBridge { static DiskReadResult poll(const std::shared_ptr<Ticket>& t) { return t->pending?DiskReadResult::Pending:DiskReadResult::Success; } };
constexpr int cudaSuccess=0;
int error=0, drains=0, resets=0;
int cudaStreamSynchronize(int) { ++drains; return error; }
const char* cudaGetErrorString(int) { return "injected hard error"; }
struct Sequence { unsigned lane=0,text_kv_valid=64,mtp_kv_valid=63; bool tail_hidden_valid=true; };
void ordered_reset(Sequence&) { ++resets; }
void commit_sequence_kv(Sequence&,int,int) {}
struct Progress { bool failing=false,done=false,success=true; unsigned phase=3,page=4; std::shared_ptr<Ticket> read; std::shared_ptr<int> state_scratch_lease; };
struct Staged { unsigned base=64,cursor=64,advertised_base=64; };
int main() {
Sequence sequence; Progress progress; Staged staged; struct { int stream=0; } device;
progress.read=std::make_shared<Ticket>(); progress.state_scratch_lease=std::make_shared<int>(1);
'''
suffix=r'''
assert(!fail()); assert(progress.failing && !progress.done);
assert(progress.read && progress.state_scratch_lease && drains==0 && resets==0);
progress.read->pending=false;
assert(fail()); assert(progress.done && !progress.success && !progress.read && !progress.state_scratch_lease);
assert(staged.base==0 && staged.cursor==0 && staged.advertised_base==0);
assert(resets==1 && drains==2 && sequence.text_kv_valid==0 && sequence.mtp_kv_valid==0);
auto pid=fork(); assert(pid>=0); if(pid==0) {error=1; fail(); std::_Exit(99);} int status; waitpid(pid,&status,0); assert(WIFEXITED(status)&&WEXITSTATUS(status)==70);
ninfer::seed_test::stage_hook=[](const char*){throw std::runtime_error("injected");};
for(auto name:{"prepare","read_submit","read_accepted","progress","before_finalize"}) { bool caught=false;try{NINFER_SEED_STAGE(name);}catch(const std::runtime_error&){caught=true;} assert(caught); }
std::puts("PASS actual fail lambda: pending borrow retains lease, terminal drain/reset, honest base=0, hard drain exit70; hook throws");
}
'''
Path('/tmp/seed_boundary.cpp').write_text(prefix+body+suffix)
subprocess.run(['g++','-std=c++20','-O2','-DNINFER_SEED_TEST_HOOKS','-I/src/src','/tmp/seed_boundary.cpp','-o','/tmp/seed_boundary'],check=True)
subprocess.run(['/tmp/seed_boundary'],check=True)
