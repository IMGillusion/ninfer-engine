#include "core/spill_copy.h"
#include <cassert>
#include <array>
#include <functional>
#include <vector>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
using namespace ninfer;
struct State {
 int fail=0, calls=0, drain_error=0, error=1, drains=0;
 bool pin=true, scratch=true, disk=false;
 std::vector<std::function<void()>> pending;
};
struct Backend {
 State* s;
 template<class F> int submit(F f) {
  assert(s->pin && s->scratch && !s->disk);
  if (++s->calls==s->fail) return s->error;
  s->pending.emplace_back(f); return 0;
 }
 int drain() {
  assert(s->pin && s->scratch && !s->disk); ++s->drains;
  if(s->drain_error) return s->drain_error;
  for(auto& f:s->pending) f(); s->pending.clear(); return 0;
 }
 bool recoverable(int e) {return e==1;}
 [[noreturn]] void fatal(int,int) {
  assert(s->pin && s->scratch && !s->disk); _exit(70);
 }
};
void run(int failure,int sync_error=0,int error=1) {
 State s; s.fail=failure;s.drain_error=sync_error;s.error=error;
 std::array<int,6> src{101,202,303,404,505,606}, dst{};
 SpillCopy copy{Backend{&s}};
 for(int i=0;i<6;++i) copy.submit([&,i]{dst[i]=src[i];});
 bool ok=copy.finish();
 assert(s.drains==1 && s.pending.empty());
 s.pin=false;
 if(ok) {s.disk=true; assert(dst==src);} else {assert(!s.disk); assert(s.calls==failure);}
 s.scratch=false;
}
int main() {
 run(0);run(1);run(3);run(6);
 for(auto [f,d,e]:{std::array<int,3>{0,9,1},{3,9,1},{3,0,9}}) {
  auto pid=fork();assert(pid>=0);if(!pid){run(f,d,e);_exit(99);}
  int status=0;assert(waitpid(pid,&status,0)==pid);assert(WIFEXITED(status)&&WEXITSTATUS(status)==70);
 }
 State s; try {SpillCopy copy{Backend{&s}};copy.submit([]{});throw std::runtime_error("after submit");}catch(...){}
 assert(s.drains==1 && s.pending.empty());
 puts("spill_copy_test PASS: first/middle/last submit, drain/hard failure exit70, payload, exception drain");
}
