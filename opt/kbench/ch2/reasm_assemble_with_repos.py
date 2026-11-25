import sys
from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CuAsmLogger import CuAsmLogger

CuAsmLogger.initLogger(log_file=None, stdout_level=25)
inp, repos, out = sys.argv[1:4]
cap = CuAsmParser()
cap.setInsAsmRepos(repos, arch='sm_61')
cap.parse(inp)
cap.saveAsCubin(out)
