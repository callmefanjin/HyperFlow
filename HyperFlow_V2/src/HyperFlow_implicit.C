/**
 * HyperFlow V2 fully coupled implicit solver.
 *
 * CVODE BDF advances the complete semi-discrete conservative system. There is
 * no Heun / reactor / Heun operator splitting in this executable.
 */

#include "cantera/core.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "implicitIntegrator.H"
#include "preComputation.H"
#include "tool_Function.H"

using namespace Cantera;
using namespace Eigen;
using namespace std;

namespace {

void validateInitialSnapshot(const StateSnapshot& snapshot,
                             const InputData& input,
                             const Cantera::ThermoPhase& gas,
                             bool require_metadata,
                             bool require_converged)
{
    if (snapshot.U.rows() != input.nVar) {
        throw runtime_error("initial HDF5 U rows do not match nVar");
    }
    if (snapshot.nVar > 0 && snapshot.nVar != input.nVar) {
        throw runtime_error("initial HDF5 nVar attribute does not match input");
    }
    if (snapshot.nSpecies > 0 && snapshot.nSpecies != input.nSpecies) {
        throw runtime_error(
            "initial HDF5 nSpecies attribute does not match mechanism");
    }
    if (static_cast<int>(snapshot.xface.size()) != snapshot.U.cols()) {
        throw runtime_error("initial HDF5 x and U dimensions do not match");
    }
    if (snapshot.xface.size() < 2) {
        throw runtime_error("initial HDF5 needs at least two axial coordinates");
    }
    if (snapshot.A_array.cols() != snapshot.U.cols()
        || snapshot.A_array.rows() != 1) {
        throw runtime_error("initial HDF5 A and U dimensions do not match");
    }
    if (!snapshot.U.allFinite() || !snapshot.xface.allFinite()
        || !snapshot.A_array.allFinite()
        || (snapshot.A_array.array() <= 0.0).any()) {
        throw runtime_error("initial HDF5 contains non-finite or non-positive data");
    }
    for (int index = 0; index + 1 < snapshot.xface.size(); ++index) {
        if (!(snapshot.xface(index + 1) > snapshot.xface(index))) {
            throw runtime_error("initial HDF5 coordinates must increase strictly");
        }
    }
    if (snapshot.grid_location != "cell_center"
        && snapshot.grid_location != "face") {
        throw runtime_error("unsupported initial HDF5 grid_location: "
                            + snapshot.grid_location);
    }
    if (!snapshot.mechanism.empty()) {
        const auto initial_mechanism =
            filesystem::path(snapshot.mechanism).filename();
        const auto requested_mechanism =
            filesystem::path(input.mechanism).filename();
        if (initial_mechanism != requested_mechanism) {
            throw runtime_error("initial HDF5 mechanism does not match input: "
                                + initial_mechanism.string() + " vs "
                                + requested_mechanism.string());
        }
    } else if (require_metadata) {
        throw runtime_error("initial HDF5 is missing mechanism metadata");
    }
    if (snapshot.species_names.empty()) {
        if (require_metadata) {
            throw runtime_error("initial HDF5 is missing species_names metadata");
        }
    } else {
        if (static_cast<int>(snapshot.species_names.size()) != input.nSpecies) {
            throw runtime_error("initial HDF5 species_names count mismatch");
        }
        for (int species = 0; species < input.nSpecies; ++species) {
            const string expected = gas.speciesName(species);
            if (snapshot.species_names[static_cast<size_t>(species)] != expected) {
                throw runtime_error(
                    "initial HDF5 species order mismatch at index "
                    + to_string(species) + ": got "
                    + snapshot.species_names[static_cast<size_t>(species)]
                    + ", expected " + expected);
            }
        }
    }
    if (require_converged
        && (snapshot.meta.kind != "steady" || !snapshot.meta.converged)) {
        throw runtime_error(
            "implicit.require_converged_initial_state is true, but the initial "
            "HDF5 is not marked as a converged steady state");
    }
    double initial_left = snapshot.xface(0);
    double initial_right = snapshot.xface(snapshot.xface.size() - 1);
    if (snapshot.grid_location == "cell_center") {
        initial_left -= 0.5 * (snapshot.xface(1) - snapshot.xface(0));
        initial_right += 0.5 * (snapshot.xface(snapshot.xface.size() - 1)
                               - snapshot.xface(snapshot.xface.size() - 2));
    }
    const double target_left = input.xface(0);
    const double target_right = input.xface(input.nface - 1);
    const double tolerance = 1.0e-9
        * max({1.0, abs(initial_right - initial_left),
               abs(target_right - target_left)});
    if (abs(initial_left - target_left) > tolerance
        || abs(initial_right - target_right) > tolerance) {
        throw runtime_error("initial and implicit domain endpoints differ");
    }
}

template<class T>
T readImplicitValue(const nlohmann::json& root,
                    const char* nested_name,
                    const char* legacy_name,
                    const T& default_value)
{
    if (root.contains("implicit")) {
        const auto& section = root.at("implicit");
        if (!section.is_object()) {
            throw runtime_error("input key 'implicit' must be a JSON object");
        }
        if (section.contains(nested_name)) {
            return section.at(nested_name).get<T>();
        }
    }
    return root.value(legacy_name, default_value);
}

IgnitionSourceOptions readIgnitionSourceOptions(
    const nlohmann::json& input_json,
    const InputData& input)
{
    IgnitionSourceOptions ignition;
    if (!input_json.contains("ignition_source")) {
        return ignition;
    }

    const auto& section = input_json.at("ignition_source");
    if (!section.is_object()) {
        throw runtime_error("input key 'ignition_source' must be a JSON object");
    }
    ignition.enabled = section.value("enabled", false);
    if (!ignition.enabled) {
        return ignition;
    }

    auto requiredDouble = [&](const char* name) {
        if (!section.contains(name)) {
            throw runtime_error(
                string("ignition_source.") + name
                + " is required when enabled");
        }
        const double value = section.at(name).get<double>();
        if (!isfinite(value)) {
            throw runtime_error(
                string("ignition_source.") + name + " must be finite");
        }
        return value;
    };

    ignition.total_energy = requiredDouble("total_energy");
    ignition.center = requiredDouble("center");
    ignition.half_length = requiredDouble("half_length");
    ignition.start_time = requiredDouble("start_time");
    ignition.duration = requiredDouble("duration");

    if (!(ignition.total_energy > 0.0)) {
        throw runtime_error("ignition_source.total_energy must be positive");
    }
    if (!(ignition.half_length > 0.0)) {
        throw runtime_error("ignition_source.half_length must be positive");
    }
    if (ignition.start_time < 0.0) {
        throw runtime_error("ignition_source.start_time must be non-negative");
    }
    if (!(ignition.duration > 0.0)) {
        throw runtime_error("ignition_source.duration must be positive");
    }

    const double left = ignition.center - ignition.half_length;
    const double right = ignition.center + ignition.half_length;
    const double domain_left = input.xface(0);
    const double domain_right = input.xface(input.nface - 1);
    const double x_tolerance =
        1.0e-12 * max(1.0, abs(domain_right - domain_left));
    if (left < domain_left - x_tolerance
        || right > domain_right + x_tolerance) {
        throw runtime_error(
            "ignition_source spatial support exceeds the domain");
    }

    const double end_time = ignition.start_time + ignition.duration;
    const double t_tolerance = 1.0e-12 * max(1.0, input.t_final);
    if (!isfinite(end_time) || end_time > input.t_final + t_tolerance) {
        throw runtime_error(
            "ignition_source pulse must finish no later than t_final");
    }

    bool found_area = false;
    for (int cell = 0; cell < input.ncell; ++cell) {
        if (input.xface(cell + 1) <= left || input.xface(cell) >= right) {
            continue;
        }
        const double candidate = input.A_array(0, cell);
        if (!found_area) {
            ignition.area = candidate;
            found_area = true;
        }
        const double area_tolerance =
            1.0e-12 * max(1.0, abs(ignition.area));
        if (abs(candidate - ignition.area) > area_tolerance
            || abs(input.A_face_array(0, cell) - ignition.area)
               > area_tolerance
            || abs(input.A_face_array(0, cell + 1) - ignition.area)
               > area_tolerance) {
            throw runtime_error(
                "ignition_source requires constant area over its full support");
        }
    }
    if (!found_area || !(ignition.area > 0.0)) {
        throw runtime_error("ignition_source did not overlap a valid cell");
    }

    ignition.peak_volumetric_power =
        2.0 * ignition.total_energy
        / (ignition.area * ignition.half_length * ignition.duration);
    if (!(ignition.peak_volumetric_power > 0.0)
        || !isfinite(ignition.peak_volumetric_power)) {
        throw runtime_error("computed ignition-source peak is invalid");
    }
    return ignition;
}

ImplicitSolverOptions readImplicitOptions(const nlohmann::json& input_json,
                                          const InputData& input)
{
    ImplicitSolverOptions options;
    options.rtol = readImplicitValue(input_json, "rtol", "implicit_rtol", 1.0e-6);
    options.atol_mass = readImplicitValue(input_json, "atol_mass", "implicit_atol_mass", 1.0e-12);
    options.atol_momentum = readImplicitValue(input_json, "atol_momentum", "implicit_atol_momentum", 1.0e-8);
    options.atol_energy = readImplicitValue(input_json, "atol_energy", "implicit_atol_energy", 1.0e-3);
    options.atol_species = readImplicitValue(input_json, "atol_species", "implicit_atol_species", 1.0e-14);
    options.initial_step = readImplicitValue(input_json, "initial_step", "implicit_initial_step", input.dt);
    options.max_step = readImplicitValue(input_json, "max_step", "implicit_max_step", 10.0 * input.dt);
    options.max_internal_steps = readImplicitValue(input_json, "max_internal_steps", "implicit_max_internal_steps", 1000000L);
    options.max_order = readImplicitValue(input_json, "max_order", "implicit_max_order", 2);
    options.krylov_dimension = readImplicitValue(input_json, "krylov_dimension", "implicit_krylov_dimension", 30);
    options.use_chemistry_preconditioner = readImplicitValue(
        input_json, "use_chemistry_preconditioner",
        "implicit_use_chemistry_preconditioner", false);
    options.enforce_species_positivity = readImplicitValue(
        input_json, "enforce_species_positivity",
        "implicit_enforce_species_positivity", true);

    if (input_json.contains("implicit")) {
        const auto& section = input_json.at("implicit");
        if (!section.contains("use_chemistry_preconditioner")
            && section.contains("use_preconditioner")) {
            options.use_chemistry_preconditioner =
                section.at("use_preconditioner").get<bool>();
        }
    }
    options.ignition_source =
        readIgnitionSourceOptions(input_json, input);
    return options;
}

} // namespace

int run(int argc, char* argv[])
{
    cout << "HyperFlow V2 fully coupled CVODE-BDF solver is running..." << endl;
    if (argc < 2) {
        throw runtime_error("Usage: HyperFlow_implicit <case_dir>");
    }

    const CasePaths case_paths = resolveCaseFiles(argv[1], kInputChemJson, false);
    cout << "Case directory: " << case_paths.case_dir << endl;
    cout << "Input JSON: " << case_paths.input_json << endl;
    ifstream input_file(case_paths.input_json);
    if (!input_file) {
        throw runtime_error("Cannot open JSON file: " + case_paths.input_json);
    }
    nlohmann::json input_json;
    input_file >> input_json;

    const string initial_h5 = resolveInitialStateH5(
        case_paths.case_dir,
        input_json.value("initial_state_h5", string()));
    cout << "Initial HDF5: " << initial_h5 << endl;

    InputData input = preComputation(input_json, false);
    auto solution = newSolution(input.mechanism);
    auto gas = solution->thermo();
    const StateSnapshot initial_snapshot = readStateHDF5(initial_h5);
    const bool require_initial_metadata = readImplicitValue(
        input_json, "require_initial_metadata",
        "implicit_require_initial_metadata", true);
    const bool require_converged_initial_state = readImplicitValue(
        input_json, "require_converged_initial_state",
        "implicit_require_converged_initial_state", false);
    validateInitialSnapshot(initial_snapshot, input, *gas,
                            require_initial_metadata,
                            require_converged_initial_state);
    cout << "Loaded initial state: kind=" << initial_snapshot.meta.kind
         << ", converged=" << initial_snapshot.meta.converged
         << ", steady_error=" << initial_snapshot.meta.steady_error
         << ", steady_residual_definition="
         << (initial_snapshot.meta.steady_residual_definition.empty()
                 ? "legacy_or_unspecified"
                 : initial_snapshot.meta.steady_residual_definition)
         << ", grid_location=" << initial_snapshot.grid_location << endl;
    if (initial_snapshot.meta.kind == "steady"
        && !initial_snapshot.meta.converged) {
        cerr << "WARNING: initial steady HDF5 is not converged; this run is "
             << "software verification only" << endl;
    }

    MatrixXd U = interpolateConservedState(
        initial_snapshot.xface, initial_snapshot.U, input.xcell);
    prepareFullyImplicitInitialState(U, input);

    const int snapshot_count = input_json.value("n_snapshots", 11);
    if (snapshot_count < 1) {
        throw runtime_error("n_snapshots must be at least one");
    }
    const int last_snapshot = snapshot_count - 1;
    auto snapshotTime = [&](int index) {
        return snapshot_count == 1
            ? input.t_final
            : index * input.t_final / static_cast<double>(last_snapshot);
    };
    auto snapshotPath = [&](int index) {
        return case_paths.case_dir + "/state_implicit_t" + to_string(index)
             + "of" + to_string(last_snapshot) + ".h5";
    };
    auto writeSnapshot = [&](int index, double time, const MatrixXd& state) {
        StateSnapshotMeta metadata;
        metadata.kind = "unsteady_implicit";
        metadata.time = time;
        metadata.converged = true;
        metadata.steady_iter = 0;
        metadata.steady_error = 0.0;
        const string path = snapshotPath(index);
        writeStateHDF5(path, metadata, input, input.xcell,
                       input.A_array, state, *gas);
        cout << "Implicit state written to " << path
             << " (t=" << time << " s)" << endl;
    };

    vector<double> output_times;
    int next_snapshot = 0;
    if (snapshot_count > 1) {
        writeSnapshot(0, 0.0, U);
        next_snapshot = 1;
    }
    for (int index = next_snapshot; index < snapshot_count; ++index) {
        output_times.push_back(snapshotTime(index));
    }

    const ImplicitSolverOptions options = readImplicitOptions(input_json, input);
    cout << "Implicit system: nVar=" << input.nVar
         << ", physical cells=" << input.ncell
         << ", ODE size="
         << static_cast<long>(input.nVar) * input.ncell << endl;
    cout << "CVODE BDF: max_order=" << options.max_order
         << ", rtol=" << options.rtol
         << ", initial_step=" << options.initial_step
         << ", max_step=" << options.max_step
         << ", SPGMR dimension=" << options.krylov_dimension
         << ", chemistry_preconditioner="
         << (options.use_chemistry_preconditioner ? "on" : "off")
         << endl;
    if (options.ignition_source.enabled) {
        const auto& ignition = options.ignition_source;
        const double source_resolution_step = ignition.duration / 20.0;
        const double pulse_max_step =
            options.max_step > 0.0
                ? min(options.max_step, source_resolution_step)
                : source_resolution_step;
        cout << "Ignition source: enabled"
             << ", E=" << ignition.total_energy << " J"
             << ", center=" << ignition.center << " m"
             << ", half_length=" << ignition.half_length << " m"
             << ", start_time=" << ignition.start_time << " s"
             << ", duration=" << ignition.duration << " s"
             << ", area=" << ignition.area << " m^2"
             << ", qdot_peak=" << ignition.peak_volumetric_power << " W/m^3"
             << ", source_peak="
             << ignition.area * ignition.peak_volumetric_power << " W/m"
             << ", pulse_max_step=" << pulse_max_step << " s" << endl;
    } else {
        cout << "Ignition source: disabled" << endl;
    }

    int callback_index = next_snapshot;
    const auto start = chrono::steady_clock::now();
    const ImplicitSolverStats stats = integrateFullyImplicit(
        U, input, solution, output_times, options,
        [&](double time, const MatrixXd& state) {
            writeSnapshot(callback_index, time, state);
            ++callback_index;
        });
    const double elapsed = chrono::duration<double>(
        chrono::steady_clock::now() - start).count();

    cout << "CVODE flag=" << stats.cvode_flag
         << ", reached_time=" << stats.reached_time
         << ", internal_steps=" << stats.internal_steps
         << ", rhs_evals=" << stats.rhs_evaluations
         << ", nonlinear_iters=" << stats.nonlinear_iterations
         << ", nonlinear_failures=" << stats.nonlinear_convergence_failures
         << ", linear_rhs_evals="
         << stats.linear_rhs_evaluations
         << ", jacobian_vector_evals="
         << stats.jacobian_vector_evaluations
         << ", jacobian_times_setups="
         << stats.jacobian_times_setups
         << ", full_rhs_calls="
         << stats.full_rhs_calls
         << ", linear_iters=" << stats.linear_iterations
         << ", linear_failures=" << stats.linear_convergence_failures
         << ", error_test_failures=" << stats.error_test_failures
         << ", prec_setups=" << stats.preconditioner_setups
         << ", prec_solves=" << stats.preconditioner_solves
         << ", max_active_prec_blocks=" << stats.max_active_preconditioner_blocks
         << ", chemistry_jacobian_builds="
         << stats.chemistry_jacobian_builds
         << ", chemistry_jacobian_wall="
         << stats.chemistry_jacobian_wall_seconds << " s"
         << ", block_factorization_wall="
         << stats.block_factorization_wall_seconds << " s"
         << ", mean_step="
         << (stats.internal_steps > 0
             ? stats.reached_time / stats.internal_steps : 0.0)
         << ", last_step=" << stats.last_step
         << ", last_order=" << stats.last_order
         << ", elapsed=" << elapsed << " s" << endl;

    if (!stats.success) {
        cerr << "Fully implicit integration failed before t_final" << endl;
        return 2;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    try {
        return run(argc, argv);
    } catch (const exception& error) {
        cerr << "HyperFlow_implicit fatal error: " << error.what() << endl;
        return 1;
    } catch (...) {
        cerr << "HyperFlow_implicit fatal error: unknown exception" << endl;
        return 1;
    }
}
