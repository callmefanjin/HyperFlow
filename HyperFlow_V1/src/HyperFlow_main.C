/**
 * HyperFlow — variable-area 1-D compressible duct flow (optional chemistry)
 *
 * Usage: HyperFlow <case_dir>
 * Requires config.json and initial.h5 (datasets x, A, U) in case_dir.
 */

#include "cantera/core.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <Eigen/Dense>

#include "preComputation.H"
#include "tool_Function.H"
#include "coreMethod.H"
#include "boundaryCondition.H"
#include "sourceTerm.H"

using namespace Eigen;
using namespace Cantera;
using namespace std;

int main(int argc, char * argv[])
{
    if (argc < 2) {
        throw runtime_error("Usage: HyperFlow <case_dir>");
    }
    const CasePaths case_paths = resolveCaseFiles(argv[1]);
    cout << "HyperFlow case: " << case_paths.case_dir << endl;
    cout << "Config: " << case_paths.config_json << endl;

    ifstream file(case_paths.config_json);
    if (!file) {
        throw runtime_error("Cannot open config: " + case_paths.config_json);
    }
    nlohmann::json j;
    file >> j;

    MatrixXd U_temp;
    InputData input = loadCase(case_paths.case_dir, j, U_temp);
    const int n_snapshots = j.value("n_snapshots", 11);
    if (n_snapshots < 1) {
        throw runtime_error("n_snapshots must be >= 1");
    }

    cout << "mechanism=" << input.mechanism
         << " chemistry=" << (input.chemistry ? "on" : "off")
         << " nface=" << input.nface
         << " nVar=" << input.nVar << endl;

    auto sol = newSolution(input.mechanism);
    auto gas = sol->thermo();
    ReactorPool pool;
    if (input.chemistry) {
        pool = init_reactor_pool(sol);
        cout << "Reactor OpenMP threads: " << pool.n_threads << endl;
    }

    U_correctY(U_temp);
    boundaryUpdate(U_temp, input, *gas);

    MatrixXd A_array = input.A_array;
    VectorXd xface = input.xface;

    auto makeOutputPath = [&](const string & filename) {
        return case_paths.out_dir + "/" + filename;
    };

    auto writeStateH5 = [&](const string & out_path, const MatrixXd & U, double sim_time) {
        StateSnapshotMeta meta;
        meta.kind = "unsteady";
        meta.time = sim_time;
        meta.converged = true;
        writeStateHDF5(out_path, meta, input, xface, A_array, U, *gas);
    };

    MatrixXd U_new(input.nVar, input.nface);
    MatrixXd dFdx_array(input.nVar, input.nface);
    MatrixXd sourceTerm_array(input.nVar, input.nface);
    MatrixXd U_1(input.nVar, input.nface);
    MatrixXd k1(input.nVar, input.nface);
    MatrixXd k2(input.nVar, input.nface);

    /** Heun flow substep: area source − flux divergence */
    auto process1_step = [&](MatrixXd U_state, double dt) -> MatrixXd {
        auto [rhoA, u, e, Y] = baseVar_From_conserVar(U_state);
        auto [pA, T, M] = PTM_From_baseVar(rhoA, u, e, Y, *gas);
        dFdx_array = dFdx_calculation(U_state, *gas, input);
        sourceTerm_array = sourceTerm_process1(pA, rhoA, u, input);
        k1 = sourceTerm_array - dFdx_array;
        U_1 = U_state + dt * k1;
        U_correctY(U_1);
        boundaryUpdate(U_1, input, *gas);

        auto [rhoA_1, u_1, e_1, Y_1] = baseVar_From_conserVar(U_1);
        auto [pA_1, T_1, M_1] = PTM_From_baseVar(rhoA_1, u_1, e_1, Y_1, *gas);
        dFdx_array = dFdx_calculation(U_1, *gas, input);
        sourceTerm_array = sourceTerm_process1(pA_1, rhoA_1, u_1, input);
        k2 = sourceTerm_array - dFdx_array;
        MatrixXd U_out = U_state + 0.5 * dt * (k1 + k2);
        U_correctY(U_out);
        boundaryUpdate(U_out, input, *gas);
        return U_out;
    };

    const double n_steps_exact = input.t_final / input.dt;
    const int n_steps = static_cast<int>(n_steps_exact);
    if (std::abs(n_steps_exact - n_steps) > 1e-9 * std::max(1.0, input.t_final)) {
        throw runtime_error("t_final must be an integer multiple of dt");
    }
    if (n_steps <= 0) {
        throw runtime_error("n_steps must be positive");
    }
    const int snap_last_index = n_snapshots - 1;

    auto snapshotTime = [&](int index) -> double {
        if (n_snapshots == 1) {
            return input.t_final;
        }
        return index * input.t_final / static_cast<double>(snap_last_index);
    };

    auto makeSnapshotFilename = [&](int index) -> string {
        return "state_t" + to_string(index) + "of" + to_string(snap_last_index) + ".h5";
    };

    cout << "Time marching: n_steps=" << n_steps
         << ", dt=" << input.dt
         << ", t_final=" << input.t_final
         << ", n_snapshots=" << n_snapshots << endl;

    const double dt = input.dt;
    const int print_interval = 500;
    int next_snapshot = 0;
    if (n_snapshots > 1) {
        const string snap_file = makeOutputPath(makeSnapshotFilename(0));
        writeStateH5(snap_file, U_temp, 0.0);
        cout << "State written to " << snap_file << " (t=0 s)" << endl;
        next_snapshot = 1;
    }

    const auto total_t0 = chrono::steady_clock::now();
    for (int i = 0; i < n_steps; i++) {
        const bool do_print = (i % print_interval == 0) || (i == n_steps - 1);
        const auto loop_t0 = chrono::steady_clock::now();
        const double sim_time = (i + 1) * input.dt;

        if (input.chemistry) {
            // Strang: C(dt/2) → chemistry(dt) → C(dt/2)
            MatrixXd U_flow = process1_step(U_temp, dt / 2);
            const auto reactor_t0 = chrono::steady_clock::now();
            reactor_calculation(U_flow, pool, input);
            const double reactor_ms = chrono::duration<double, milli>(
                chrono::steady_clock::now() - reactor_t0).count();
            U_correctY(U_flow);
            boundaryUpdate(U_flow, input, *gas);
            U_new = process1_step(U_flow, dt / 2);

            if (do_print) {
                MatrixXd residual = (U_new - U_temp).array().abs() / (U_temp.array() + 1e-10);
                const double loop_ms = chrono::duration<double, milli>(
                    chrono::steady_clock::now() - loop_t0).count();
                cout << "step: " << i << " time: " << sim_time
                     << " residual: " << residual.maxCoeff()
                     << " reactor: " << reactor_ms << " ms"
                     << " loop: " << loop_ms << " ms" << endl;
            }
        } else {
            U_new = process1_step(U_temp, dt);
            if (do_print) {
                MatrixXd residual = (U_new - U_temp).array().abs() / (U_temp.array() + 1e-10);
                const double loop_ms = chrono::duration<double, milli>(
                    chrono::steady_clock::now() - loop_t0).count();
                cout << "step: " << i << " time: " << sim_time
                     << " residual: " << residual.maxCoeff()
                     << " loop: " << loop_ms << " ms" << endl;
            }
        }

        U_temp = U_new;

        while (next_snapshot < n_snapshots) {
            const double t_snap = snapshotTime(next_snapshot);
            if (sim_time + 1e-14 < t_snap) {
                break;
            }
            const string snap_file = makeOutputPath(makeSnapshotFilename(next_snapshot));
            writeStateH5(snap_file, U_temp, sim_time);
            cout << "State written to " << snap_file
                 << " (t=" << sim_time << " s)" << endl;
            ++next_snapshot;
        }
    }
    while (next_snapshot < n_snapshots) {
        const double t_snap = snapshotTime(next_snapshot);
        const string snap_file = makeOutputPath(makeSnapshotFilename(next_snapshot));
        writeStateH5(snap_file, U_temp, t_snap);
        cout << "State written to " << snap_file
             << " (t=" << t_snap << " s)" << endl;
        ++next_snapshot;
    }

    const double total_sec = chrono::duration<double>(
        chrono::steady_clock::now() - total_t0).count();
    cout << "Total elapsed: " << total_sec << " s ("
         << total_sec / 60.0 << " min)" << endl;
    return 0;
}
