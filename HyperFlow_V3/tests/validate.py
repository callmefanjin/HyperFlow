#!/usr/bin/env python3
"""Numerical checks; no claim of a converged detonation reproduction."""
import copy
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cantera as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get('HYPERFLOW_BUILD_DIR', ROOT/'build')) / 'hyperflow2d'
DEST = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT / ('validation_' + time.strftime('%Y%m%d_%H%M%S'))
DEST.mkdir(parents=True, exist_ok=True)
BASE = dict(problem='uniform', base_cells=[16, 16], length=[.01, .01],
            max_level=2, blocking_factor=4, max_grid_size=16, tag_buffer=1,
            tag_mode='box', tag_box=[.003, .007, .003, .007], periodic=[1, 1],
            chemistry=False, chemistry_threads=8, cfl=.6, end_time=2e-7,
            regrid_interval=2, mechanism=str(ROOT/'H2_Li.yaml'),
            write_plotfiles=False)
REPORT = dict(directory=str(DEST), tests={})


def run(name, config):
    config = copy.deepcopy(config)
    src = DEST / (name + '.json')
    output = DEST / name
    if (output/'summary.json').exists():
        if json.loads(src.read_text()) != config:
            raise RuntimeError('resume input differs: '+name)
        result = json.loads((output/'summary.json').read_text())
        if not result['completed']:
            raise RuntimeError('cannot resume incomplete result: '+name)
        print('verified existing', name, flush=True)
        return output, result
    src.write_text(json.dumps(config, indent=2) + '\n')
    with (DEST / (name + '.log')).open('w') as log:
        result = subprocess.run([str(EXE), str(src), str(output)], stdout=log,
                                stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError(f'{name}: see {DEST / (name + ".log")}')
    print('completed', name, flush=True)
    return output, json.loads((output / 'summary.json').read_text())


def rows(path, frame=-1):
    paths = sorted(path.glob('state_*.csv'), key=lambda p: int(p.stem.split('_')[1]))
    return np.genfromtxt(paths[frame], names=True, delimiter=',')


def conserved_error(summary):
    a = np.array(summary['initial_conserved_integrals'])
    b = np.array(summary['final_conserved_integrals'])
    mass = np.sum(a[:9])
    return dict(mass=abs(np.sum(b[:9])-mass)/mass,
                energy=abs(b[11]-a[11])/max(abs(a[11]), 1e-100),
                remap=summary['max_remap_relative_defect'])


def check_geometry(data, config):
    level = data['level'].astype(int)
    base_dx = np.array(config['length']) / config['base_cells']
    assert np.allclose(data['dx'], base_dx[0] / 2.**level, rtol=1e-13, atol=0)
    assert np.allclose(data['dy'], base_dx[1] / 2.**level, rtol=1e-13, atol=0)
    assert abs(np.sum(data['dx']*data['dy']) / np.prod(config['length'])-1) < 1e-13
    full = np.zeros(np.array(config['base_cells']) * 2**config['max_level'], dtype=np.int8)
    for q in data:
        r = 2**(config['max_level']-int(q['level']))
        i, j = int(q['i'])*r, int(q['j'])*r
        full[i:i+r, j:j+r] += 1
    assert np.all(full == 1), 'leaf cells overlap or leave a hole'
    return dict(levels=np.unique(level).tolist(), leaf_cells=len(data),
                full_finest_cells=int(full.size), area=float(np.sum(data['dx']*data['dy'])))


def raster(data, config, field):
    cells = np.array(config['base_cells']) * 2**config['max_level']
    result = np.zeros(cells)
    for q in data:
        r = 2**(config['max_level']-int(q['level']))
        i, j = int(q['i'])*r, int(q['j'])*r
        result[i:i+r, j:j+r] = q[field]
    return result


try:
    config = copy.deepcopy(BASE)
    output, summary = run('uniform_local_amr', config)
    data = rows(output)
    geometry = check_geometry(data, config)
    deviation = max(float(np.ptp(data[n])/max(abs(data[n][0]), 1.))
                    for n in ('rho', 'u', 'v', 'p', 'T', 'rhoE'))
    assert deviation < 1e-12
    assert geometry['leaf_cells'] < geometry['full_finest_cells']
    assert summary['reflux_stage_interfaces'] > 0
    assert summary['retries'] == 0
    REPORT['tests']['uniform_local_amr'] = dict(**geometry, uniform_deviation=deviation,
                                              conservation=conserved_error(summary))

    config = copy.deepcopy(BASE)
    config.update(problem='entropy_wave', center=[.003, .004], sigma=.0007,
                  tag_box=[.002, .004, .003, .005], tag_velocity=[300., 200.],
                  u0=300., v0=200., end_time=4e-6, output_times=[2e-6, 4e-6],
                  regrid_interval=4)
    output, summary = run('moving_refine_coarsen', config)
    data = rows(output)
    geometry = check_geometry(data, config)
    before, after = raster(rows(output, 0), config, 'level'), raster(data, config, 'level')
    refined, coarsened = int(np.sum(after > before)), int(np.sum(after < before))
    error = conserved_error(summary)
    assert refined > 0 and coarsened > 0
    assert error['mass'] < 1e-11 and error['energy'] < 1e-11 and error['remap'] < 1e-9
    assert summary['retries'] == 0
    REPORT['tests']['moving_refine_coarsen'] = dict(**geometry, refined_pixels=refined,
                 coarsened_pixels=coarsened, conservation=error, steps=summary['steps'])

    config = copy.deepcopy(BASE)
    config.update(problem='blast2d', base_cells=[24, 24], length=[.02, .02],
                  center=[.01, .01], radius=.002, periodic=[0, 0],
                  boundary_lo=['reflecting', 'reflecting'], boundary_hi=['reflecting', 'reflecting'],
                  tag_mode='physics', pressure_tag=.06, temperature_tag=.06,
                  end_time=2e-6, output_times=[1e-6, 2e-6], regrid_interval=4,
                  write_plotfiles=True)
    config.pop('tag_box')
    output, summary = run('blast2d_inert', config)
    data = rows(output)
    geometry = check_geometry(data, config)
    error = conserved_error(summary)
    assert error['mass'] < 1e-11 and error['energy'] < 1e-11
    assert summary['retries'] == 0
    image = raster(data, config, 'p')
    symmetry = float(np.mean(np.abs(image-image.T))/np.mean(image))
    REPORT['tests']['blast2d_inert'] = dict(**geometry, conservation=error,
                                          transpose_pressure_error=symmetry, steps=summary['steps'])

    config = copy.deepcopy(BASE)
    config.update(problem='uniform', T0=1500., P0=5*101325., chemistry=True,
                  end_time=2e-7, chemistry_threads=8)
    output, summary = run('uniform_chemistry_omp8', config)
    data = rows(output)
    g = ct.Solution(config['mechanism'])
    g.TPX = config['T0'], config['P0'], 'H2:2,O2:1,N2:3.76'
    reactor = ct.IdealGasReactor(g, energy='on')
    network = ct.ReactorNet([reactor]); network.rtol=1e-10; network.atol=1e-18
    network.advance(config['end_time'])
    reference = reactor.thermo
    y = np.array([np.mean(data['Y_'+n]) for n in g.species_names])
    t_error = float(abs(np.mean(data['T'])-reference.T)/reference.T)
    y_error = float(np.max(abs(y-reference.Y)))
    assert t_error < 1e-7 and y_error < 1e-8
    assert summary['last_chemistry_team'] == 8
    assert summary['chemistry_cell_calls'] == len(data)*summary['steps']
    error = conserved_error(summary)
    assert error['mass'] < 1e-12 and error['energy'] < 1e-12
    REPORT['tests']['zero_dimensional_cvode'] = dict(relative_temperature_error=t_error,
        max_absolute_Y_error=y_error, conservation=error, chemistry_cell_calls=summary['chemistry_cell_calls'],
        expected_leaf_calls=len(data)*summary['steps'], last_team=summary['last_chemistry_team'])

    config['chemistry_threads'] = 1
    one, one_summary = run('uniform_chemistry_omp1', config)
    a, b = rows(one), data
    order_a, order_b = np.lexsort((a['y'], a['x'])), np.lexsort((b['y'], b['x']))
    absolute_error = {n:float(np.max(abs(a[n][order_a]-b[n][order_b]))) for n in a.dtype.names}
    thread_error = max(absolute_error[n]/max(float(np.max(abs(a[n]))),1.) for n in a.dtype.names)
    assert thread_error < 1e-12
    REPORT['tests']['chemistry_parallel'] = dict(max_normalized_difference=thread_error,
                                               absolute_differences=absolute_error)

    config = copy.deepcopy(BASE)
    config.update(problem='blast2d', chemistry=True, max_level=1, center=[.005, .005],
                  radius=.0015, tag_mode='physics', periodic=[0, 0],
                  boundary_hi=['reflecting', 'reflecting'], end_time=2e-7,
                  output_times=[1e-7, 2e-7], regrid_interval=2)
    config.pop('tag_box')
    output, summary = run('blast2d_reacting', config)
    data = rows(output)
    error = conserved_error(summary)
    assert error['mass'] < 1e-11 and error['energy'] < 1e-11
    y_fields = [n for n in data.dtype.names if n.startswith('Y_')]
    Y = np.column_stack([data[n] for n in y_fields])
    assert np.min(Y) >= 0 and np.max(abs(Y.sum(axis=1)-1)) < 1e-12
    assert np.isfinite(data['T']).all() and np.min(data['T']) > 0
    radical_growth = float(np.max(data['Y_H']+data['Y_O']+data['Y_OH']))
    assert radical_growth > 1e-12
    assert summary['retries'] == 0
    REPORT['tests']['blast2d_reacting'] = dict(conservation=error,
        peak_radical_mass_fraction=radical_growth, steps=summary['steps'], retries=summary['retries'],
        first_order_stages=summary['positivity_first_order_stages'])

    config['chemistry_threads']=1
    one, _ = run('blast2d_reacting_omp1', config)
    a, b = rows(one), data
    oa, ob = np.lexsort((a['y'],a['x'])), np.lexsort((b['y'],b['x']))
    assert len(a)==len(b)
    parallel_errors={n:float(np.max(abs(a[n][oa]-b[n][ob])))/max(float(np.max(abs(a[n]))),1.)
                     for n in a.dtype.names}
    assert max(parallel_errors.values()) < 1e-10
    REPORT['tests']['nonuniform_reacting_parallel']=dict(normalized_max_errors=parallel_errors)

    # The original uniform solver is read/executed only; all outputs stay here.
    legacy = Path(os.environ.get('HYPERFLOW_LEGACY_EXE', '/nonexistent/optional-legacy-baseline'))
    if legacy.exists():
        old_case = dict(case='example6', nx=40, ny=8, end_time=1e-6, cfl=.6,
                        snapshots=1, chemistry_threads=8, mechanism=BASE['mechanism'])
        old_input = DEST/'legacy_uniform.json'
        if not (DEST/'legacy_uniform').exists():
            old_input.write_text(json.dumps(old_case, indent=2))
            with (DEST/'legacy_uniform.log').open('w') as log:
                subprocess.run([str(legacy), str(old_input), str(DEST/'legacy_uniform')],
                               stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        config = copy.deepcopy(BASE)
        config.update(problem='example6', base_cells=[40, 8], length=[4., 1.],
                      max_level=0, periodic=[0, 0], chemistry=True,
                      boundary_lo=['transmissive', 'reflecting'],
                      boundary_hi=['transmissive', 'reflecting'], end_time=1e-6)
        output, summary = run('uniform_legacy_parity', config)
        a, b = rows(DEST/'legacy_uniform'), rows(output)
        oa, ob = np.lexsort((a['y'], a['x'])), np.lexsort((b['y'], b['x']))
        fields = ['rho', 'u', 'v', 'p', 'T']+[n for n in b.dtype.names if n.startswith('Y_')]
        errors = {n:float(np.max(abs(a[n][oa]-b[n][ob]))/max(np.max(abs(a[n])), 1.)) for n in fields}
        assert max(errors.values()) < 1e-10
        REPORT['tests']['uniform_legacy_parity'] = dict(normalized_max_errors=errors)

    REPORT['passed'] = True
except Exception as error:
    REPORT['passed'] = False
    REPORT['error'] = repr(error)
    raise
finally:
    REPORT['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'src').glob('*') if p.is_file()}
    (DEST/'report.json').write_text(json.dumps(REPORT, indent=2)+'\n')
    print('REPORT', DEST/'report.json', flush=True)
