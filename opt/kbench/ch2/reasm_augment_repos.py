import sys
import time
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuAsmLogger import CuAsmLogger

CuAsmLogger.initLogger(log_file=None, stdout_level=25)
t0=time.time()
repos = CuInsAssemblerRepos(arch='sm_61')
repos.setToDefaultInsAsmDict()
inp = sys.argv[1] if len(sys.argv) > 1 else 'kernels_unpatched.sass'
out = sys.argv[2] if len(sys.argv) > 2 else 'Repos.sm_61.production.txt'
f = CuInsFeeder(inp, archfilter='sm_61')
n = repos.update(f)
print('updated new', n, 'keys', len(repos), 'elapsed', time.time()-t0, flush=True)
repos.save2file(out)
# Verify that every production instruction now re-encodes exactly.
ok = repos.verify(CuInsFeeder(inp, archfilter='sm_61'))
print('verify', ok, 'elapsed', time.time()-t0, flush=True)
raise SystemExit(0 if ok else 2)
