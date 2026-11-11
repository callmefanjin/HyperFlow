#!/usr/bin/env python3
"""Check both case configurations through the same executable."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import cantera as ct

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
parser.add_argument('--reference-teng',type=Path)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
report=dict(passed=False,solver='hyperflow2d',cases={})

def run(c,dest,exe=None):
    src=dest.with_suffix('.json');src.write_text(json.dumps(c,indent=2)+'\n')
    with dest.with_suffix('.log').open('w') as log:
        subprocess.run([str((exe or args.build_dir/'hyperflow2d').resolve()),str(src.resolve()),str(dest.resolve())],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,
                       env=dict(os.environ,OMP_NUM_THREADS=str(c['chemistry_threads']),OMP_DYNAMIC='FALSE'))
    snaps=sorted(dest.glob('state_*.csv'),key=lambda f:int(f.stem.split('_')[-1]))
    return (np.genfromtxt(snaps[0],names=True,delimiter=','),
            np.genfromtxt(snaps[-1],names=True,delimiter=','),json.loads((dest/'summary.json').read_text()))

def compare(a,b):
    assert len(a)==len(b)
    oa,ob=np.lexsort((a['y'],a['x'])),np.lexsort((b['y'],b['x']))
    fields=['rho','u','v','p','T','rhoE']+[n for n in a.dtype.names if n.startswith('Y_')]
    errors={n:float(np.max(abs(a[n][oa]-b[n][ob]))/max(float(np.max(abs(a[n]))),1.)) for n in fields}
    assert max(errors.values())<1e-10,errors
    return max(errors.values())

try:
    for case in ['sun_cj','teng_case1']:
        c=json.loads((ROOT/'cases'/case/'smoke.json').read_text());c['mechanism']=str(ROOT/'H2_Li.yaml')
        initial,final,s=run(c,args.output/case)
        assert s['completed'] and s['retries']==0
        assert abs(s['final_time']-c['end_time'])<1e-20
        assert all(np.isfinite(final[n]).all() for n in final.dtype.names)
        Y=np.column_stack([final[n] for n in final.dtype.names if n.startswith('Y_')])
        assert np.min(Y)>=0 and np.max(abs(Y.sum(axis=1)-1))<1e-12
        assert all(np.min(final[n])>0 for n in ['rho','p','T'])
        serial=copy.deepcopy(c);serial['chemistry_threads']=1
        _,one,_=run(serial,args.output/(case+'_omp1'))
        result=dict(final_time_s=s['final_time'],steps=s['steps'],initial_leaf_cells=len(initial),
                    final_leaf_cells=len(final),dx_min_m=s['dx_min'],dy_min_m=s['dy_min'],
                    retries=s['retries'],max_normalized_thread_difference=compare(final,one))
        if case=='sun_cj':
            assert abs(s['dx_min']-20e-6)<1e-16
            assert np.max(abs(initial['u']))==0 and np.max(abs(initial['v']))==0
            volume=initial['dx']*initial['dy'];hot=np.isclose(initial['T'],3000.,rtol=1e-10)
            assert abs(volume.sum()-.5)<1e-13 and abs(volume[hot].sum()-.002)<1e-14
            g=ct.Solution(str(ROOT/'H2_Li.yaml'))
            def integrals(T,P,V):
                g.TPX=T,P,'H2:2,O2:1,N2:3.76'
                return g.density*V,g.density*g.int_energy_mass*V
            fresh=integrals(300.,101325.,.498);spot=integrals(3000.,90*101325.,.002)
            a=np.array(s['initial_conserved_integrals']);b=np.array(s['final_conserved_integrals'])
            expected_mass=fresh[0]+spot[0];expected_energy=fresh[1]+spot[1]
            assert abs(a[:9].sum()/expected_mass-1)<1e-12 and abs(a[11]/expected_energy-1)<1e-12
            mass_error=float(abs(b[:9].sum()/a[:9].sum()-1));energy_error=float(abs(b[11]/a[11]-1))
            assert mass_error<1e-11 and energy_error<1e-9
            # Compare rows against each field's global scale. A local velocity
            # near zero must not be treated as the characteristic flow speed.
            fields=['rho','u','v','p','T','rhoE']+[n for n in final.dtype.names if n.startswith('Y_')]
            deviation=0.
            scales={n:max(float(np.max(abs(final[n]))),1.) for n in fields}
            absolute={n:0. for n in fields}
            order=np.lexsort((final['i'],final['level']));f=final[order]
            cuts=np.flatnonzero((np.diff(f['i'])!=0)|(np.diff(f['level'])!=0))+1
            for group in np.split(f,cuts):
                for n in fields:
                    absolute[n]=max(absolute[n],float(np.ptp(group[n])))
                    deviation=max(deviation,float(np.ptp(group[n]))/scales[n])
            assert deviation<1e-8
            result.update(initial_hotspot_volume_m3=float(volume[hot].sum()),mass_relative_change=mass_error,
                          energy_relative_change=energy_error,max_transverse_normalized_deviation=deviation,
                          max_transverse_absolute_deviations=absolute)
        else:
            assert abs(s['dx_min']-4e-6)<1e-16 and abs(s['dy_min']-4e-6)<1e-16
            if args.reference_teng:
                _,old,_=run(c,args.output/'teng_original',args.reference_teng)
                result['max_normalized_original_difference']=compare(final,old)
        report['cases'][case]=result
        print('PASS same solver:',case,flush=True)
    report['passed']=True
except Exception as e:
    report['error']=repr(e)
    raise
finally:
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
