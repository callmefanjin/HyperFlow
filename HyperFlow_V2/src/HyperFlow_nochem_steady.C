/**
 * HyperFlow_nochem_steady — 无化学稳态预迭代，输出 HDF5 初始场
 */

#include "cantera/core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "preComputation.H"
#include "tool_Function.H"
#include "coreMethod.H"
#include "boundaryCondition.H"
#include "timeIntegrator.H"
using namespace Eigen;
using namespace Cantera;
using namespace std;

namespace
{
constexpr int kResidualCheckInterval = 100;
constexpr const char kResidualDefinition[] = "dimensionless_semi_discrete_rhs_v1";

struct ResidualPeak
{
    int variable = -1;
    int cell = -1;
    double signed_value = 0.0;
};

struct SteadyResidual
{
    double linf = 0.0;
    double l2 = 0.0;
    double mass_linf = 0.0;
    double momentum_linf = 0.0;
    double energy_linf = 0.0;
    double species_linf = 0.0;
    MatrixXd scaled_rhs;
    ResidualPeak overall_peak;
    ResidualPeak mass_peak;
    ResidualPeak momentum_peak;
    ResidualPeak energy_peak;
    ResidualPeak species_peak;
};

ResidualPeak locateResidualPeak(const MatrixXd & scaled_rhs,
                                int first_variable,
                                int variable_count)
{
    ResidualPeak peak;
    if (variable_count <= 0) {
        return peak;
    }
    Eigen::Index local_variable = 0;
    Eigen::Index cell = 0;
    scaled_rhs.block(first_variable, 0, variable_count, scaled_rhs.cols())
        .cwiseAbs().maxCoeff(&local_variable, &cell);
    peak.variable = first_variable + static_cast<int>(local_variable);
    peak.cell = static_cast<int>(cell);
    peak.signed_value = scaled_rhs(peak.variable, peak.cell);
    return peak;
}

string residualVariableName(int variable, const ThermoPhase & gas)
{
    if (variable == 0) {
        return "mass";
    } else if (variable == 1) {
        return "momentum";
    } else if (variable == 2) {
        return "energy";
    } else if (variable >= 3) {
        return "species:" + gas.speciesName(static_cast<size_t>(variable - 3));
    }
    return "none";
}

VectorXd makeResidualScales(const InputData & input,
                            ThermoPhase & gas,
                            double & reference_time)
{
    gas.setState_TPX(input.temperature_inlet,
                     input.pressure_inlet,
                     input.airMoleFraction);
    const double sound_speed = gas.soundSpeed();
    const double velocity_scale = std::max(std::abs(input.velocity_inlet),
                                           sound_speed);
    const double mass_state_scale = input.density_inlet * input.area_inlet;
    const double total_energy_inlet = input.internal_energy_inlet
                                    + 0.5 * input.velocity_inlet * input.velocity_inlet;
    const double energy_scale = std::max(std::abs(total_energy_inlet),
                                         sound_speed * sound_speed);
    const double domain_length = input.xface(input.nface - 1) - input.xface(0);
    reference_time = domain_length
                   / (std::abs(input.velocity_inlet) + sound_speed);

    if (!(sound_speed > 0.0) || !std::isfinite(sound_speed)
        || !(mass_state_scale > 0.0) || !std::isfinite(mass_state_scale)
        || !(domain_length > 0.0) || !std::isfinite(domain_length)
        || !(reference_time > 0.0) || !std::isfinite(reference_time)) {
        throw runtime_error("invalid reference scales for steady residual");
    }

    VectorXd scales(input.nVar);
    scales(0) = mass_state_scale;
    scales(1) = mass_state_scale * velocity_scale;
    scales(2) = mass_state_scale * energy_scale;
    const double active_injection_massflow =
        input.injection_enabled ? input.massflow_injection : 0.0;
    const double outlet_massflow =
        input.massflow_inlet + active_injection_massflow;
    for (int k = 0; k < input.nSpecies; ++k) {
        // A species absent at the inlet can still be introduced by injection.
        // Scale it with its expected fully mixed outlet mass fraction instead
        // of the numerical floor alone, otherwise roundoff dominates Linf.
        const double outlet_species_fraction =
            (input.massflow_inlet * input.Y_inlet(k, 0)
             + active_injection_massflow * input.Y_injection(k, 0))
            / outlet_massflow;
        scales(3 + k) = mass_state_scale
                      * std::max({std::abs(input.Y_inlet(k, 0)),
                                  std::abs(outlet_species_fraction), 1e-12});
    }
    if (!scales.allFinite() || (scales.array() <= 0.0).any()) {
        throw runtime_error("non-positive or non-finite steady residual scale");
    }
    return scales;
}

SteadyResidual evaluateSteadyResidual(const MatrixXd & U,
                                      ThermoPhase & gas,
                                      const InputData & input,
                                      FlowWorkBuffers & buf,
                                      const VectorXd & scales,
                                      double reference_time,
                                      int spatial_order)
{
    // The convergence test is based on the actual semi-discrete equation
    // dU/dt = R(U), not on a time-step-dependent state increment.
    flow_rhs(U, gas, input, input.injection_enabled, buf, spatial_order);
    MatrixXd scaled_rhs = buf.rhs;
    for (int variable = 0; variable < input.nVar; ++variable) {
        scaled_rhs.row(variable) *= reference_time / scales(variable);
    }

    SteadyResidual residual;
    residual.scaled_rhs = scaled_rhs;
    residual.overall_peak = locateResidualPeak(scaled_rhs, 0, input.nVar);
    residual.mass_peak = locateResidualPeak(scaled_rhs, 0, 1);
    residual.momentum_peak = locateResidualPeak(scaled_rhs, 1, 1);
    residual.energy_peak = locateResidualPeak(scaled_rhs, 2, 1);
    residual.linf = std::abs(residual.overall_peak.signed_value);
    residual.l2 = std::sqrt(residual.scaled_rhs.squaredNorm()
                          / static_cast<double>(residual.scaled_rhs.size()));
    residual.mass_linf = std::abs(residual.mass_peak.signed_value);
    residual.momentum_linf = std::abs(residual.momentum_peak.signed_value);
    residual.energy_linf = std::abs(residual.energy_peak.signed_value);
    if (input.nSpecies > 0) {
        residual.species_peak = locateResidualPeak(
            scaled_rhs, 3, input.nSpecies);
        residual.species_linf = std::abs(residual.species_peak.signed_value);
    }
    return residual;
}

void printResidualPeaks(const char * label,
                        const SteadyResidual & residual,
                        const InputData & input,
                        const ThermoPhase & gas)
{
    const auto print_peak = [&](const char * component, const ResidualPeak & peak) {
        if (peak.cell < 0 || peak.variable < 0) {
            return;
        }
        cout << "  " << component
             << ": variable=" << residualVariableName(peak.variable, gas)
             << " cell=" << peak.cell
             << " x=" << input.xcell(peak.cell)
             << " signed=" << peak.signed_value << endl;
    };
    cout << label << " residual peak locations:" << endl;
    print_peak("overall", residual.overall_peak);
    print_peak("mass", residual.mass_peak);
    print_peak("momentum", residual.momentum_peak);
    print_peak("energy", residual.energy_peak);
    print_peak("species", residual.species_peak);
}
} // namespace

int run(int argc, char * argv[])
{
    cout << "HyperFlow_nochem_steady is running..." << endl;
    if (argc < 2) {
        throw runtime_error("Usage: HyperFlow_nochem_steady <case_dir>");
    }
    const CasePaths case_paths = resolveCaseFiles(argv[1], kInputNochemJson, false);
    cout << "Case directory: " << case_paths.case_dir << endl;
    cout << "Input JSON: " << case_paths.input_json << endl;

    ifstream file(case_paths.input_json);
    if (!file) {
        throw runtime_error("Cannot open JSON file: " + case_paths.input_json);
    }
    nlohmann::json j;
    file >> j;
    // Corrected no-chemistry defaults. Explicitly written choices still win,
    // while old reacting-flow JSON files retain their legacy defaults.
    if (!j.contains("friction_energy_model")) {
        j["friction_energy_model"] = "adiabatic";
    }
    if (!j.contains("reconstruction_variables")) {
        j["reconstruction_variables"] = "primitive";
    }

    /*预计算（稳态网格）*/
    InputData input = preComputation(j, true);
    cout << "Steady grid: steady_cell=" << input.steady_cell
         << " (ncell=" << input.ncell << ", nface=" << input.nface << ")" << endl;
    cout << "friction_energy_model=" << input.friction_energy_model
         << ", reconstruction_variables=" << input.reconstruction_variables
         << endl;
    MatrixXd A_array = input.A_array;
    VectorXd xcell = input.xcell;
    auto sol = newSolution(input.mechanism);
    auto gas = sol->thermo();

    /*下面开始创建计算初始条件*/
    // 初场 u 取入口速度全场均匀（不用 u=ṁ/(ρA)）：伪瞬态更稳定，第一步 Cantera 不易崩溃。
    // 初场不要求 ṁ 沿程守恒；稳态迭代收敛后的流场满足 mdot 守恒即可。
    MatrixXd rhoA_array = input.density_inlet * A_array.array();
    MatrixXd u_array = input.velocity_inlet * MatrixXd::Ones(1, input.ncell);
    MatrixXd e_array = input.internal_energy_inlet * MatrixXd::Ones(1, input.ncell);
    MatrixXd et_array = e_array.array() + 0.5 * u_array.array().square();
    MatrixXd Y_array = input.Y_inlet.replicate(1, input.ncell);

    // 创建守恒变量数组（避免 MatrixXd = expr.array() 与 operator<< 得到 0×0）
    MatrixXd U_temp(input.nVar, input.ncell);
    U_temp.row(0) = rhoA_array;
    U_temp.row(1) = (rhoA_array.array() * u_array.array()).matrix();
    U_temp.row(2) = (rhoA_array.array() * et_array.array()).matrix();
    U_temp.block(3, 0, input.nSpecies, input.ncell) =
        (Y_array.array().rowwise() * rhoA_array.row(0).array()).matrix();

    cout << "steady error_tolerance=" << input.error_tolerance
         << ", steady_iter_max=" << input.steady_iter_max
         << endl;
    const int steady_spatial_order = j.value("steady_spatial_order", 2);
    if (steady_spatial_order != 1 && steady_spatial_order != 2) {
        throw runtime_error(
            "steady_spatial_order must be 1 (piecewise constant) or 2 (NND)");
    }
    cout << "steady_spatial_order=" << steady_spatial_order << endl;

    /*稳态预迭代*/
    FlowWorkBuffers step_buf(input.nVar, input.ncell);
    FlowWorkBuffers residual_buf(input.nVar, input.ncell);
    /*无化学纯流动稳态迭代*/
    double steady_dt = 1e-7;
    const double error_tolerance = input.error_tolerance;
    const int steady_iter_max = input.steady_iter_max;
    double CFL_max = input.cfl_allowance;
    int steady_iter = 0;
    double reference_time = 0.0;
    const VectorXd residual_scales = makeResidualScales(input, *gas, reference_time);
    SteadyResidual residual = evaluateSteadyResidual(
        U_temp, *gas, input, residual_buf, residual_scales, reference_time,
        steady_spatial_order);
    const double initial_residual_linf = residual.linf;
    double error = residual.linf;
    cout << "Steady residual definition: " << kResidualDefinition
         << ", reference_time=" << reference_time << " s"
         << ", check_interval=" << kResidualCheckInterval << endl;
    cout << "initial residual: Linf=" << residual.linf
         << " L2=" << residual.l2
         << " (mass=" << residual.mass_linf
         << " momentum=" << residual.momentum_linf
         << " energy=" << residual.energy_linf
         << " species=" << residual.species_linf << ")" << endl;
    printResidualPeaks("initial", residual, input, *gas);
    const auto steady_t0 = chrono::steady_clock::now();
    while (error > error_tolerance && steady_iter < steady_iter_max) {
        const auto step_t0 = chrono::steady_clock::now();

        // 基于 CFL 数计算时间步长
        auto[rhoA_temp, u_temp, e_temp, Y_temp] = baseVar_From_conserVar(U_temp);
        double lambda_max = 0.0;
        for (int cell = 0; cell < input.ncell; ++cell) {
            gas->setMassFractions(Y_temp.col(cell).data());
            gas->setState_UV(e_temp(0, cell), 1.0 / rhoA_temp(0, cell));
            lambda_max = std::max(
                lambda_max, std::abs(u_temp(0, cell)) + gas->soundSpeed());
        }
        if (!(lambda_max > 0.0) || !std::isfinite(lambda_max)) {
            throw runtime_error("non-positive or non-finite maximum wave speed");
        }
        const double dx_min = (input.xface.tail(input.nface - 1)
                     - input.xface.head(input.nface - 1)).minCoeff();
        steady_dt = CFL_max * dx_min / lambda_max;

        // 三阶 SSP-RK3
        SSPRK3_step(U_temp, *gas, input, steady_dt, input.injection_enabled,
                    step_buf, steady_spatial_order);
        U_temp = step_buf.U_out;

        const double step_ms = chrono::duration<double, milli>(
            chrono::steady_clock::now() - step_t0).count();
        ++steady_iter;
        const bool residual_check = steady_iter % kResidualCheckInterval == 0
                                 || steady_iter == steady_iter_max;
        if (residual_check) {
            residual = evaluateSteadyResidual(
                U_temp, *gas, input, residual_buf, residual_scales,
                reference_time, steady_spatial_order);
            error = residual.linf;
        }
        if (residual_check
            && (steady_iter % 1000 == 0 || error <= error_tolerance
                || steady_iter == steady_iter_max)) {
            cout << "steady iter: " << steady_iter
                 << " residual Linf: " << residual.linf
                 << " L2: " << residual.l2
                 << " (mass=" << residual.mass_linf
                 << " momentum=" << residual.momentum_linf
                 << " energy=" << residual.energy_linf
                 << " species=" << residual.species_linf << ")"
                 << " peak="
                 << residualVariableName(residual.overall_peak.variable, *gas)
                 << "@x=" << input.xcell(residual.overall_peak.cell)
                 << " dt: " << steady_dt << " elapsed: " << step_ms << " ms" << endl;
        }
    }
    const double steady_sec = chrono::duration<double>(
        chrono::steady_clock::now() - steady_t0).count();
    const bool converged = (error <= error_tolerance);
    if (converged) {
        cout << "Steady state converged at error: " << error
             << ", elapsed: " << steady_sec << " s ("
             << steady_sec / 60.0 << " min)" << endl;
    } else {
        cerr << "WARNING: steady NOT converged after " << steady_iter
             << " iterations, error=" << error
             << ", elapsed: " << steady_sec << " s" << endl;
    }
    printResidualPeaks("final", residual, input, *gas);

    StateSnapshotMeta meta;
    meta.kind = "steady";
    meta.time = 0.0;
    meta.converged = converged;
    meta.steady_iter = steady_iter;
    meta.steady_error = error;
    meta.steady_residual_definition = kResidualDefinition;
    meta.steady_residual_l2 = residual.l2;
    meta.steady_residual_initial_linf = initial_residual_linf;
    meta.steady_residual_reduction = initial_residual_linf > 0.0
        ? residual.linf / initial_residual_linf
        : (residual.linf == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    meta.steady_residual_mass_linf = residual.mass_linf;
    meta.steady_residual_momentum_linf = residual.momentum_linf;
    meta.steady_residual_energy_linf = residual.energy_linf;
    meta.steady_residual_species_linf = residual.species_linf;
    meta.steady_reference_time = reference_time;
    meta.steady_spatial_order = steady_spatial_order;

    writeStateHDF5(case_paths.steady_h5, meta, input, xcell, A_array, U_temp, *gas);
    appendSteadyResidualHDF5(
        case_paths.steady_h5,
        residual.scaled_rhs,
        xcell,
        residual.overall_peak.variable,
        residual.overall_peak.cell,
        residualVariableName(residual.overall_peak.variable, *gas));
    cout << "Steady state written to " << case_paths.steady_h5 << endl;

    writeInputDataJSON(case_paths.computed_input_json, input);
    cout << "Computed input written to " << case_paths.computed_input_json << endl;

    return converged ? 0 : 1;
}

int main(int argc, char* argv[])
{
    try {
        return run(argc, argv);
    } catch (const exception& error) {
        cerr << "HyperFlow_nochem_steady fatal error: " << error.what() << endl;
        return 2;
    } catch (...) {
        cerr << "HyperFlow_nochem_steady fatal error: unknown exception" << endl;
        return 2;
    }
}
