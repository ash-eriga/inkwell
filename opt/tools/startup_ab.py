#!/usr/bin/env python3
"""Interleaved startup A/B harness. Run under opt/tools/gpu-run -p normal."""
import json, os, re, subprocess, sys, time, hashlib, threading
from pathlib import Path

outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('opt/startup-ab')
outdir.mkdir(parents=True, exist_ok=True)
ctl = os.environ.get('STARTUP_CTL', '/tmp/inkwell-baseline-302339f')
cand = os.environ.get('STARTUP_CAND', 'target/release/inkwell')
rounds = int(os.environ.get('STARTUP_ROUNDS', '3'))
model = os.path.expanduser('~/models/inkling')
base_env = os.environ.copy()
base_env.update({
    'GGML_PASCAL_GRAPHS': '1',
    'INKWELL_MAX_CTX': '512',
    'INKWELL_CACHE_MARGIN_MB': '10',
    'INKWELL_EXL3_CACHE_K': '20',
    'INKWELL_MOE_V2': '1',
    'INKWELL_MOE_NKY': '2',
    'INKWELL_GENS': '64',
    'INKWELL_STEPDBG': '1',
    'INKWELL_LOADPROF': '1',
})
if (Path.home()/'lfm-gtx1060-bench/cuda-12.6/lib64').exists():
    base_env['LD_LIBRARY_PATH'] = str(Path.home()/'lfm-gtx1060-bench/cuda-12.6/lib64') + ':' + str(Path.home()/'lfm-gtx1060-bench/build-inkwell-dp4a/bin') + ':' + base_env.get('LD_LIBRARY_PATH','')

def vram_peak(proc, stop, out):
    peak = 0
    while proc.poll() is None and not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2)
            for line in r.stdout.splitlines():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) == 2 and parts[1].isdigit() and int(parts[0]) == proc.pid:
                    peak = max(peak, int(parts[1]))
        except Exception:
            pass
        time.sleep(0.15)
    out['vram_mib'] = peak

def one(label, binary):
    env = base_env.copy()
    logp = outdir / f'{label}.log'
    start = time.monotonic()
    with logp.open('w') as log:
        proc = subprocess.Popen([binary, model, '--run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        stop = threading.Event(); vr = {}
        vt = threading.Thread(target=vram_peak, args=(proc, stop, vr), daemon=True); vt.start()
        first = None; lines = []
        assert proc.stderr is not None
        for line in proc.stderr:
            lines.append(line); log.write(line)
            if first is None and 'stepwall=' in line:
                first = time.monotonic()
        rc = proc.wait(); stop.set(); vt.join(timeout=1)
        if proc.stdout is not None: log.write(proc.stdout.read())
    end = time.monotonic()
    txt = ''.join(lines)
    genline = next((x for x in lines if 'generated text' in x), '')
    md5 = hashlib.md5(genline.encode()).hexdigest()[:8] if genline else ''
    m = re.search(r'decode 8 tok in ([\d.]+)(ms|s)', txt)
    ms_tok = None
    if m:
        sec = float(m.group(1)) / 1000 if m.group(2) == 'ms' else float(m.group(1))
        ms_tok = sec * 1000 / 64
    marks = dict((a, float(b)) for a, b in re.findall(r'loadprof@ (\S+) ([\d.]+)', txt))
    phases = [(a, float(b), int(c)) for a, b, c in re.findall(r'loadprof: (\S+)\s+([\d.]+)ms n=(\d+)', txt)]
    return {'label': label, 'binary': binary, 'rc': rc, 'startup_s': (first-start) if first else None,
            'wall_s': end-start, 'md5': md5, 'ms_tok': ms_tok, 'vram_mib': vr.get('vram_mib'),
            'load_ms': marks.get('load_st_done'), 'gen0_ms': marks.get('gen0_step_done'), 'phases': phases}

results=[]
for r in range(rounds):
    # ctl,cand,cand,ctl controls drift while keeping the pairing compact
    order=[('ctl',ctl),('cand',cand)] if r%2==0 else [('cand',cand),('ctl',ctl)]
    for kind,b in order:
        res=one(f'r{r}-{kind}',b); results.append(res)
        print('RUN '+json.dumps({k:v for k,v in res.items() if k!='phases'},sort_keys=True),flush=True)
(outdir/'results.json').write_text(json.dumps(results,indent=2))
print('BATCH DONE',flush=True)
