#!/usr/bin/env python3
"""Select a case configuration for the common HyperFlow-2D executable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('case',choices=['sun_cj','teng_case1'])
parser.add_argument('output',type=Path)
parser.add_argument('--input',type=Path)
parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
args=parser.parse_args()
out=args.output.resolve()
if out.exists():parser.error('output directory already exists')
exe=args.build_dir.resolve()/'hyperflow2d'
if not exe.is_file():parser.error('build/hyperflow2d is required')
c=json.loads((args.input or ROOT/'cases'/args.case/'input.json').read_text())
if c['problem']!={'sun_cj':'sun_cj','teng_case1':'teng_osw'}[args.case]:parser.error('case and problem mismatch')
path=Path(c['mechanism']);c['mechanism']=str((ROOT/path if not path.is_absolute() else path).resolve(strict=True))
out.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='hyperflow2d-') as temp:
    src=Path(temp)/'input.json';src.write_text(json.dumps(c,indent=2)+'\n')
    p=subprocess.run([str(exe),str(src),str(out)],env=dict(os.environ,OMP_NUM_THREADS=str(c['chemistry_threads']),OMP_DYNAMIC='FALSE'))
raise SystemExit(p.returncode)
