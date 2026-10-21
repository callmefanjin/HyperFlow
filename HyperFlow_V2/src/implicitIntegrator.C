#include "implicitIntegrator.H"

#include "boundaryCondition.H"
#include "coreMethod.H"
#include "timeIntegrator.H"

#include <cantera/base/Solution.h>
#include <cantera/kinetics/Kinetics.h>
#include <cantera/thermo/ThermoPhase.h>

#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

class ChemistryBlockFactor
{
public:
    explicit ChemistryBlockFactor(int size) : size_(size) {}
    bool compute(const MatrixXd& block)
    {
        if (block.rows() != size_ || block.cols() != size_ || !block.allFinite()) return false;
        factor_.compute(block);
        return factor_.matrixLU().allFinite()
            && factor_.matrixLU().diagonal().cwiseAbs().minCoeff()
               > std::numeric_limits<double>::min();
    }
    VectorXd solve(const VectorXd& rhs) { return factor_.solve(rhs); }
private:
    int size_;
    Eigen::PartialPivLU<MatrixXd> factor_;
};

int dominantInletSpecies(const InputData& input)
{
    int dependent_species = 0;
    for (int species = 1; species < input.nSpecies; ++species) {
        if (input.Y_inlet(species, 0)
            > input.Y_inlet(dependent_species, 0)) {
            dependent_species = species;
        }
    }
    return dependent_species;
}

VectorXd buildIgnitionSpatialAverage(
    const InputData& input,
    const IgnitionSourceOptions& ignition)
{
    VectorXd spatial_average = VectorXd::Zero(input.ncell);
    if (!ignition.enabled) {
        return spatial_average;
    }

    const double left_support = ignition.center - ignition.half_length;
    const double right_support = ignition.center + ignition.half_length;
    auto antiderivative = [&](double x) {
        const double xi = x - ignition.center;
        return 0.5 * xi
             + ignition.half_length / (2.0 * M_PI)
             * std::sin(M_PI * xi / ignition.half_length);
    };

    double spatial_integral = 0.0;
    for (int cell = 0; cell < input.ncell; ++cell) {
        const double cell_left = input.xface(cell);
        const double cell_right = input.xface(cell + 1);
        const double overlap_left = std::max(cell_left, left_support);
        const double overlap_right = std::min(cell_right, right_support);
        if (!(overlap_right > overlap_left)) {
            continue;
        }
        const double integral =
            antiderivative(overlap_right) - antiderivative(overlap_left);
        const double dx = cell_right - cell_left;
        spatial_average(cell) = integral / dx;
        spatial_integral += integral;
    }

    const double integral_tolerance =
        1.0e-11 * std::max(1.0, ignition.half_length);
    if (std::abs(spatial_integral - ignition.half_length)
        > integral_tolerance) {
        throw std::runtime_error(
            "ignition-source discrete spatial integral is inconsistent");
    }
    return spatial_average;
}

double ignitionTimeProfile(double time, const IgnitionSourceOptions& ignition)
{
    if (!ignition.enabled
        || time <= ignition.start_time
        || time >= ignition.start_time + ignition.duration) {
        return 0.0;
    }
    const double phase =
        M_PI * (time - ignition.start_time) / ignition.duration;
    const double sine = std::sin(phase);
    return sine * sine;
}

struct ImplicitUserData
{
    InputData* input;
    std::shared_ptr<Cantera::Solution> flow_solution;
    Cantera::ThermoPhase* flow_gas;
    FlowWorkBuffers flow_buffers;

    MatrixXd U_work;
    VectorXd ignition_spatial_average;

    std::vector<std::shared_ptr<Cantera::Solution>> chemistry_solutions;
    std::string last_error;
    int dependent_species;
    const ImplicitSolverOptions* options;
    std::vector<MatrixXd> chemistry_jacobians;

    std::vector<std::unique_ptr<ChemistryBlockFactor>> chemistry_factors;
    std::vector<unsigned char> chemistry_block_active;
    std::vector<unsigned char> preconditioner_block_active;

    long max_active_preconditioner_blocks = 0;

    long chemistry_jacobian_builds = 0;
    double chemistry_jacobian_wall_seconds = 0.0;
    double block_factorization_wall_seconds = 0.0;
    void* cvode_memory = nullptr;
    long rhs_call_count = 0;

    double max_rhs_time = 0.0;

    std::chrono::steady_clock::time_point progress_start =
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_progress_report =
        progress_start;

    ImplicitUserData(InputData& input_in,
                     std::shared_ptr<Cantera::Solution> solution,
                     const MatrixXd& initial_state,
                     const ImplicitSolverOptions& options_in)
        : input(&input_in)
        , flow_solution(std::move(solution))
        , flow_gas(flow_solution->thermo().get())
        , flow_buffers(input_in.nVar, input_in.ncell)
        , U_work(initial_state)
        , ignition_spatial_average(
              buildIgnitionSpatialAverage(input_in, options_in.ignition_source))

        , dependent_species(dominantInletSpecies(input_in))
        , options(&options_in)
    {

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        chemistry_solutions.reserve(static_cast<size_t>(nthreads));
        for (int thread = 0; thread < nthreads; ++thread) {
            chemistry_solutions.push_back(Cantera::newSolution(input_in.mechanism));
        }
        if (options_in.use_chemistry_preconditioner) {
            chemistry_jacobians.assign(
                static_cast<size_t>(nInterior()),
                MatrixXd::Zero(input_in.nVar, input_in.nVar));
            chemistry_block_active.assign(
                static_cast<size_t>(nInterior()), 0);
            preconditioner_block_active.assign(
                static_cast<size_t>(nInterior()), 0);
            chemistry_factors.reserve(static_cast<size_t>(nInterior()));
            for (int q = 0; q < nInterior(); ++q) {
                chemistry_factors.push_back(
                    std::make_unique<ChemistryBlockFactor>(
                        input_in.nVar));
            }
        }

    }

    int nInterior() const { return input->ncell; }

    sunindextype vectorSize() const
    {
        return static_cast<sunindextype>(nInterior()) * input->nVar;
    }
};

sunindextype stateIndex(int interior_index, int variable, int nvar)
{
    return static_cast<sunindextype>(interior_index) * nvar + variable;
}

void repairInitialSpeciesRoundoff(MatrixXd& U,
                                  const InputData& input,
                                  int dependent_species)
{
    for (int i = 0; i < input.ncell; ++i) {
        const double rhoA = U(0, i);
        if (!(rhoA > 0.0) || !std::isfinite(rhoA)) {
            throw std::runtime_error(
                "invalid total mass while repairing initial species roundoff");
        }
        const double tolerance = 1.0e-12 * rhoA;
        double independent_sum = 0.0;
        for (int species = 0; species < input.nSpecies; ++species) {
            if (species == dependent_species) {
                continue;
            }
            double& rhoAY = U(3 + species, i);
            if (!std::isfinite(rhoAY) || rhoAY < -tolerance) {
                throw std::runtime_error(
                    "initial state contains a materially negative species");
            }
            if (rhoAY < 0.0) {
                rhoAY = 0.0;
            }
            independent_sum += rhoAY;
        }
        double dependent_mass = rhoA - independent_sum;
        if (dependent_mass < -tolerance || !std::isfinite(dependent_mass)) {
            throw std::runtime_error(
                "initial independent species exceed total mass");
        }
        U(3 + dependent_species, i) = std::max(0.0, dependent_mass);
    }
}

void validateImplicitOptions(const ImplicitSolverOptions& options)
{
    auto require_positive_finite = [](double value, const char* name) {
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(std::string(name)
                                     + " must be positive and finite");
        }
    };
    require_positive_finite(options.rtol, "implicit rtol");
    require_positive_finite(options.atol_mass, "implicit mass atol");
    require_positive_finite(options.atol_momentum, "implicit momentum atol");
    require_positive_finite(options.atol_energy, "implicit energy atol");
    require_positive_finite(options.atol_species, "implicit species atol");
    if (options.initial_step < 0.0 || !std::isfinite(options.initial_step)) {
        throw std::runtime_error(
            "implicit initial_step must be non-negative and finite");
    }
    if (options.max_step < 0.0 || !std::isfinite(options.max_step)) {
        throw std::runtime_error(
            "implicit max_step must be non-negative and finite");
    }
    if (options.initial_step > 0.0 && options.max_step > 0.0
        && options.initial_step > options.max_step) {
        throw std::runtime_error(
            "implicit initial_step must not exceed max_step");
    }
    if (options.max_internal_steps <= 0) {
        throw std::runtime_error("implicit max_internal_steps must be positive");
    }
    if (options.max_order < 1 || options.max_order > 5) {
        throw std::runtime_error("implicit max_order must be in [1, 5]");
    }
    if (options.krylov_dimension <= 0) {
        throw std::runtime_error("implicit krylov_dimension must be positive");
    }

    const auto& ignition = options.ignition_source;
    if (ignition.enabled) {
        require_positive_finite(
            ignition.total_energy, "ignition_source total_energy");
        require_positive_finite(
            ignition.half_length, "ignition_source half_length");
        require_positive_finite(
            ignition.duration, "ignition_source duration");
        require_positive_finite(ignition.area, "ignition_source area");
        require_positive_finite(
            ignition.peak_volumetric_power,
            "ignition_source peak volumetric power");
        if (!std::isfinite(ignition.center)) {
            throw std::runtime_error("ignition_source center must be finite");
        }
        if (ignition.start_time < 0.0
            || !std::isfinite(ignition.start_time)) {
            throw std::runtime_error(
                "ignition_source start_time must be non-negative and finite");
        }
    }
}

double effectiveMaximumStep(const ImplicitSolverOptions& options)
{
    double maximum_step = options.max_step;
    const auto& ignition = options.ignition_source;
    if (ignition.enabled) {
        const double source_resolution_step = ignition.duration / 20.0;
        if (!(maximum_step > 0.0) || maximum_step > source_resolution_step) {
            maximum_step = source_resolution_step;
        }
    }
    return maximum_step;
}

void unpackInterior(N_Vector y, ImplicitUserData& data)
{
    const sunrealtype* values = N_VGetArrayPointer(y);
    const int nvar = data.input->nVar;
    for (int q = 0; q < data.nInterior(); ++q) {
        const int i = q;
        for (int variable = 0; variable < nvar; ++variable) {
            data.U_work(variable, i) = values[stateIndex(q, variable, nvar)];
        }
    }
}

void packInterior(const MatrixXd& U, N_Vector y, const InputData& input)
{
    sunrealtype* values = N_VGetArrayPointer(y);
    for (int q = 0; q < input.ncell; ++q) {
        const int i = q;
        for (int variable = 0; variable < input.nVar; ++variable) {
            values[stateIndex(q, variable, input.nVar)] = U(variable, i);
        }
    }
}

bool localChemistrySource(const double* state,
                          int spatial_index,
                          const InputData& input,
                            const ImplicitSolverOptions&,
                          Cantera::Solution& solution,
                          VectorXd& source,
                          std::string* failure_reason = nullptr,
                          double* evaluated_temperature = nullptr)
{
    auto fail = [&](const std::string& reason) {
        if (failure_reason != nullptr) {
            *failure_reason = reason;
        }
        return false;
    };
    source.setZero(input.nVar);
    const double rhoA = state[0];
    const double area = input.A_array(0, spatial_index);
    if (!(rhoA > 0.0) || !(area > 0.0)) {
        return fail("invalid rhoA or area");
    }

    const double velocity = state[1] / rhoA;
    const double internal_energy = state[2] / rhoA
                                 - 0.5 * velocity * velocity;
    const double density = rhoA / area;
    if (!(density > 0.0) || !std::isfinite(internal_energy)) {
        return fail("invalid density or internal energy");
    }

    const double eta = input.fullmix_flag
        ? 1.0
        : std::clamp(input.mix_by_time(0, spatial_index), 0.0, 1.0);
    std::vector<double> mixed_mass(static_cast<size_t>(input.nSpecies), 0.0);
    double rhoA_mixed = 0.0;
    double rhoA_unmixed_fuel = 0.0;
    for (int species = 0; species < input.nSpecies; ++species) {
        double rhoAY = state[3 + species];
        if (!std::isfinite(rhoAY)) {
            return fail("non-finite species state");
        }
        // Preserve a differentiable RHS for CVODE Newton/Krylov trial states.
        // CVODE constraints still keep accepted species states non-negative.
        const bool fuel_species = input.Y_injection(species, 0) > 0.0;
        const double participating = fuel_species ? eta * rhoAY : rhoAY;
        mixed_mass[static_cast<size_t>(species)] = participating;
        rhoA_mixed += participating;
        if (fuel_species) {
            rhoA_unmixed_fuel += (1.0 - eta) * rhoAY;
        }
    }
    if (!(rhoA_mixed > 0.0) || !std::isfinite(rhoA_mixed)) {
        return fail("invalid participating mixed mass");
    }

    for (double& value : mixed_mass) {
        value /= rhoA_mixed;
    }
    // Remove the internal energy carried by the unmixed fuel before forming
    // the thermodynamic state of the participating mixture. Using total-cell
    // energy with a composition that excludes unmixed DME gives an artificial
    // sub-physical temperature because the fuel formation energy is retained.
    const double mixed_internal_energy =
        (rhoA * internal_energy - rhoA_unmixed_fuel * input.e_fuel)
        / rhoA_mixed;
    if (!std::isfinite(mixed_internal_energy)) {
        return fail("non-finite participating mixture internal energy");
    }
    std::vector<double> wdot(static_cast<size_t>(input.nSpecies), 0.0);
    std::vector<double> molecular_weights(
        static_cast<size_t>(input.nSpecies), 0.0);
    double temperature = std::numeric_limits<double>::quiet_NaN();

    auto& gas = *solution.thermo();
    auto kinetics = solution.kinetics();
    gas.setMassFractions_NoNorm(mixed_mass.data());
    gas.setState_UV(mixed_internal_energy, 1.0 / density);
    temperature = gas.temperature();

    kinetics->getNetProductionRates(wdot.data());

    const auto& cantera_weights = gas.molecularWeights();
    molecular_weights.assign(
        cantera_weights.begin(), cantera_weights.end());

    if (evaluated_temperature != nullptr) {
        *evaluated_temperature = temperature;
    }
    const double beta = rhoA_mixed / rhoA;
    for (int species = 0; species < input.nSpecies; ++species) {
        source(3 + species) = area * beta
            * molecular_weights[static_cast<size_t>(species)]
            * wdot[static_cast<size_t>(species)];
        if (!std::isfinite(source(3 + species))) {
            return fail("non-finite chemistry source for species "
                        + std::to_string(species));
        }
    }
    // Chemistry does not directly change total mass, momentum or total energy.
    return true;
}

int fullyCoupledRhs(sunrealtype time,
                    N_Vector y,
                    N_Vector ydot,
                    void* user_data)
{
    auto& data = *static_cast<ImplicitUserData*>(user_data);
    InputData& input = *data.input;
    try {
        ++data.rhs_call_count;

        data.max_rhs_time =
            std::max(data.max_rhs_time, static_cast<double>(time));
        const auto progress_now = std::chrono::steady_clock::now();
        const double since_report = std::chrono::duration<double>(
            progress_now - data.last_progress_report).count();
        if (since_report >= 10.0) {
            const double elapsed = std::chrono::duration<double>(
                progress_now - data.progress_start).count();
            long internal_steps = 0;
            long nonlinear_failures = 0;
            long linear_iterations = 0;
            long linear_failures = 0;
            long error_test_failures = 0;
            double accepted_time = 0.0;
            double last_step = 0.0;
            if (data.cvode_memory != nullptr) {
                CVodeGetNumSteps(data.cvode_memory, &internal_steps);
                CVodeGetNumNonlinSolvConvFails(
                    data.cvode_memory, &nonlinear_failures);
                CVodeGetNumLinIters(
                    data.cvode_memory, &linear_iterations);
                CVodeGetNumLinConvFails(
                    data.cvode_memory, &linear_failures);
                CVodeGetNumErrTestFails(
                    data.cvode_memory, &error_test_failures);
                CVodeGetCurrentTime(data.cvode_memory, &accepted_time);
                CVodeGetLastStep(data.cvode_memory, &last_step);
            }
            std::cerr << "CVODE internal progress: wall=" << elapsed
                      << " s, rhs_calls=" << data.rhs_call_count
                      << ", rhs_time=" << static_cast<double>(time)
                      << " s, max_rhs_time=" << data.max_rhs_time
                      << " s, accepted_time=" << accepted_time
                      << " s, last_step=" << last_step
                      << " s, steps=" << internal_steps
                      << ", nonlinear_failures=" << nonlinear_failures
                      << ", linear_iterations=" << linear_iterations
                      << ", linear_failures=" << linear_failures
                      << ", error_test_failures=" << error_test_failures
                      << ", last_error="
                      << (data.last_error.empty() ? "none" : data.last_error)
                      << std::endl;
            data.last_progress_report = progress_now;
        }
        unpackInterior(y, data);
        update_MassFlow_by_time(static_cast<double>(time), input);
        update_mix_by_time(static_cast<double>(time), input);

        try {
            flow_rhs(data.U_work, *data.flow_gas, input, true,
                     data.flow_buffers, 2);
        } catch (const std::exception& error) {
            int minimum_energy_cell = -1;
            int maximum_density_cell = -1;
            double minimum_internal_energy =
                std::numeric_limits<double>::infinity();
            double maximum_density = 0.0;
            double minimum_rhoA =
                std::numeric_limits<double>::infinity();
            for (int cell = 0; cell < input.ncell; ++cell) {
                const double rhoA = data.U_work(0, cell);
                minimum_rhoA = std::min(minimum_rhoA, rhoA);
                if (!(rhoA > 0.0) || !std::isfinite(rhoA)) {
                    continue;
                }
                const double velocity = data.U_work(1, cell) / rhoA;
                const double internal_energy =
                    data.U_work(2, cell) / rhoA
                    - 0.5 * velocity * velocity;
                const double density =
                    rhoA / input.A_array(0, cell);
                if (internal_energy < minimum_internal_energy) {
                    minimum_internal_energy = internal_energy;
                    minimum_energy_cell = cell;
                }
                if (density > maximum_density) {
                    maximum_density = density;
                    maximum_density_cell = cell;
                }
            }
            std::ostringstream detail;
            detail << "flow RHS failure at t="
                   << static_cast<double>(time)
                   << " s, minimum_rhoA=" << minimum_rhoA
                   << ", min_e_cell=" << minimum_energy_cell
                   << ", min_e=" << minimum_internal_energy
                   << ", max_rho_cell=" << maximum_density_cell
                   << ", max_rho=" << maximum_density;
            if (minimum_energy_cell >= 0) {
                const int cell = minimum_energy_cell;
                const double rhoA = data.U_work(0, cell);
                const double velocity =
                    data.U_work(1, cell) / rhoA;
                double species_sum = 0.0;
                double minimum_species =
                    std::numeric_limits<double>::infinity();
                for (int species = 0;
                     species < input.nSpecies; ++species) {
                    const double rhoAY =
                        data.U_work(3 + species, cell);
                    species_sum += rhoAY;
                    minimum_species =
                        std::min(minimum_species, rhoAY);
                }
                detail << ", x=" << input.xcell(cell)
                       << ", rhoA=" << rhoA
                       << ", momentum=" << data.U_work(1, cell)
                       << ", total_energy=" << data.U_work(2, cell)
                       << ", velocity=" << velocity
                       << ", min_rhoAY=" << minimum_species
                       << ", sum_rhoAY-rhoA="
                       << (species_sum - rhoA);
            }
            detail << ", reason=" << error.what();
            data.last_error = detail.str();
            return 1;
        }

        const auto& ignition = data.options->ignition_source;
        const double ignition_time_profile =
            ignitionTimeProfile(static_cast<double>(time), ignition);
        if (ignition_time_profile > 0.0) {
            const double peak_source_per_length =
                ignition.area * ignition.peak_volumetric_power;
            data.flow_buffers.rhs.row(2).array() +=
                peak_source_per_length * ignition_time_profile
                * data.ignition_spatial_average.transpose().array();
        }

        std::atomic<int> chemistry_failed{0};
        int failed_cell = -1;
        std::string failed_reason;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < input.ncell; ++i) {
            if (chemistry_failed.load(std::memory_order_relaxed) != 0) {
                continue;
            }
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            try {
                VectorXd source(input.nVar);
                std::string local_failure;
                if (!localChemistrySource(data.U_work.col(i).data(), i, input,
                                          *data.options,
                                          *data.chemistry_solutions[static_cast<size_t>(thread)],
                                          source, &local_failure)) {
#pragma omp critical(implicit_chemistry_failure)
                    {
                        if (failed_cell < 0) {
                            failed_cell = i;
                            failed_reason = local_failure;
                        }
                    }
                    chemistry_failed.store(1, std::memory_order_relaxed);
                    continue;
                }
                data.flow_buffers.rhs.col(i) += source;
            } catch (const std::exception& error) {
#pragma omp critical(implicit_chemistry_failure)
                {
                    if (failed_cell < 0) {
                        failed_cell = i;
                        failed_reason = error.what();
                    }
                }
                chemistry_failed.store(1, std::memory_order_relaxed);
            } catch (...) {
#pragma omp critical(implicit_chemistry_failure)
                {
                    if (failed_cell < 0) {
                        failed_cell = i;
                        failed_reason = "unknown chemistry exception";
                    }
                }
                chemistry_failed.store(1, std::memory_order_relaxed);
            }
        }
        if (chemistry_failed.load(std::memory_order_relaxed) != 0) {
            std::ostringstream detail;
            detail << "chemistry failure at t=" << static_cast<double>(time)
                   << " s";
            if (failed_cell >= 0) {
                const double rhoA = data.U_work(0, failed_cell);
                const double momentum = data.U_work(1, failed_cell);
                const double total_energy = data.U_work(2, failed_cell);
                double min_rhoAY = std::numeric_limits<double>::infinity();
                double sum_rhoAY = 0.0;
                for (int species = 0; species < input.nSpecies; ++species) {
                    const double rhoAY =
                        data.U_work(3 + species, failed_cell);
                    min_rhoAY = std::min(min_rhoAY, rhoAY);
                    sum_rhoAY += rhoAY;
                }
                detail << ", cell=" << failed_cell
                       << ", x=" << input.xcell(failed_cell)
                       << ", rhoA=" << rhoA
                       << ", min_rhoAY=" << min_rhoAY
                       << ", sum_rhoAY-rhoA=" << (sum_rhoAY - rhoA);
                if (rhoA > 0.0) {
                    const double velocity = momentum / rhoA;
                    const double internal_energy =
                        total_energy / rhoA - 0.5 * velocity * velocity;
                    detail << ", u=" << velocity
                           << ", e=" << internal_energy;
                }
            }
            if (!failed_reason.empty()) {
                detail << ", reason=" << failed_reason;
            }
            data.last_error = detail.str();
            return 1;
        }

        // The total-mass row and all species rows are linearly dependent.
        // Independent reconstructions do not preserve that identity
        // exactly, so use the dominant inlet bath gas as the dependent row.
        // This enforces d(rhoA - sum_k rhoAY_k)/dt = 0 in the semi-discrete
        // system without projecting the state inside CVODE's Newton RHS.
        for (int i = 0; i < input.ncell; ++i) {
            double independent_species_rhs = 0.0;
            for (int species = 0; species < input.nSpecies; ++species) {
                if (species != data.dependent_species) {
                    independent_species_rhs
                        += data.flow_buffers.rhs(3 + species, i);
                }
            }
            data.flow_buffers.rhs(3 + data.dependent_species, i)
                = data.flow_buffers.rhs(0, i) - independent_species_rhs;
        }

        sunrealtype* rhs = N_VGetArrayPointer(ydot);
        for (int q = 0; q < data.nInterior(); ++q) {
            const int i = q;
            for (int variable = 0; variable < input.nVar; ++variable) {
                const double value = data.flow_buffers.rhs(variable, i);
                if (!std::isfinite(value)) {
                    data.last_error = "non-finite fully coupled RHS";
                    return 1;
                }
                rhs[stateIndex(q, variable, input.nVar)] = value;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        data.last_error = error.what();
        return 1;
    } catch (...) {
        data.last_error = "unknown fully coupled RHS failure";
        return 1;
    }
}

void enforceDependentChemistrySource(VectorXd& source,
                                     const InputData& input,
                                     int dependent_species)
{
    double independent_source = 0.0;
    for (int species = 0; species < input.nSpecies; ++species) {
        if (species != dependent_species) {
            independent_source += source(3 + species);
        }
    }
    source(3 + dependent_species) = -independent_source;
}

bool cellNeedsChemistryPreconditioner(const double* state,
                                      int spatial_index,
                                      const InputData& input)
{
    const double rhoA = state[0];
    if (!(rhoA > 0.0)) {
        return false;
    }
    const double eta = input.fullmix_flag
        ? 1.0
        : std::clamp(input.mix_by_time(0, spatial_index), 0.0, 1.0);
    if (!(eta > 1.0e-14)) {
        return false;
    }
    // Hot products and radical pools remain stiff after the fuel is depleted.
    // Keep their chemistry blocks active whenever the mixture participates.
    return true;
}

double absoluteToleranceForVariable(int variable,
                                    const ImplicitSolverOptions& options)
{
    if (variable == 0) return options.atol_mass;
    if (variable == 1) return options.atol_momentum;
    if (variable == 2) return options.atol_energy;
    return options.atol_species;
}

double finiteDifferenceIncrement(double value,
                                 int variable,
                                 const ImplicitSolverOptions& options)
{
    const double machine_epsilon = std::numeric_limits<double>::epsilon();
    const double relative_tolerance = std::max(options.rtol, machine_epsilon);
    const double reference = absoluteToleranceForVariable(variable, options)
                           / relative_tolerance;
    const double scale = std::max({std::abs(value), reference, 1.0e-30});
    return std::sqrt(relative_tolerance) * scale;
}

int chemistryPreconditionerSetup(sunrealtype time,
                                 N_Vector y,
                                 N_Vector,
                                 sunbooleantype jok,
                                 sunbooleantype* jcur_ptr,
                                 sunrealtype gamma,
                                 void* user_data)
{
    auto& data = *static_cast<ImplicitUserData*>(user_data);
    InputData& input = *data.input;
    try {
        unpackInterior(y, data);
        update_mix_by_time(static_cast<double>(time), input);
        const bool rebuild_jacobian = (jok == SUNFALSE);
        std::atomic<int> setup_failed{0};
        const auto jacobian_build_start =
            std::chrono::steady_clock::now();

#pragma omp parallel for schedule(static)
        for (int q = 0; q < data.nInterior(); ++q) {
            if (setup_failed.load(std::memory_order_relaxed) != 0) {
                continue;
            }
            const int i = q;
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            try {
                MatrixXd& jacobian =
                    data.chemistry_jacobians[static_cast<size_t>(q)];
                if (rebuild_jacobian) {
                    jacobian.setZero();
                    const bool active = data.options->use_chemistry_preconditioner
                        && cellNeedsChemistryPreconditioner(
                            data.U_work.col(i).data(), i, input);
                    data.chemistry_block_active[static_cast<size_t>(q)] =
                        active ? 1 : 0;
                    if (active) {
                        auto& solution = *data.chemistry_solutions[
                            static_cast<size_t>(thread)];

                        VectorXd base_source(input.nVar);
                        if (!localChemistrySource(
                                data.U_work.col(i).data(), i, input,
                                *data.options, solution, base_source)) {
                            setup_failed.store(
                                1, std::memory_order_relaxed);
                            continue;
                        }
                        enforceDependentChemistrySource(
                            base_source, input, data.dependent_species);

                        VectorXd perturbed_state = data.U_work.col(i);
                        VectorXd perturbed_source(input.nVar);
                        for (int variable = 0;
                             variable < input.nVar; ++variable) {
                            const double original =
                                perturbed_state(variable);
                            const double increment =
                                finiteDifferenceIncrement(
                                    original, variable, *data.options);
                            double used_increment = increment;
                            perturbed_state(variable) =
                                original + increment;
                            bool evaluated = localChemistrySource(
                                perturbed_state.data(), i, input,
                                *data.options, solution,
                                perturbed_source);
                            if (!evaluated) {
                                used_increment = -increment;
                                perturbed_state(variable) =
                                    original - increment;
                                evaluated = localChemistrySource(
                                    perturbed_state.data(), i, input,
                                    *data.options, solution,
                                    perturbed_source);
                            }
                            perturbed_state(variable) = original;
                            if (!evaluated) {
                                jacobian.col(variable).setZero();
                                continue;
                            }
                            enforceDependentChemistrySource(
                                perturbed_source, input,
                                data.dependent_species);
                            jacobian.col(variable) =
                                (perturbed_source - base_source)
                                / used_increment;
                        }

                        if (!jacobian.allFinite()) {
                            setup_failed.store(
                                1, std::memory_order_relaxed);
                            continue;
                        }
                    }
                }

            } catch (...) {
                setup_failed.store(1, std::memory_order_relaxed);
            }
        }

        if (setup_failed.load(std::memory_order_relaxed) != 0) {
            data.last_error = "block preconditioner state setup failed";
            return 1;
        }

        if (rebuild_jacobian) {
            long active_chemistry_blocks = 0;
            for (unsigned char active : data.chemistry_block_active) {
                active_chemistry_blocks += active != 0 ? 1 : 0;
            }
            data.chemistry_jacobian_builds += active_chemistry_blocks;
            data.chemistry_jacobian_wall_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - jacobian_build_start).count();
        }
        const auto factorization_start =
            std::chrono::steady_clock::now();
        long active_blocks = 0;
        const MatrixXd identity =
            MatrixXd::Identity(input.nVar, input.nVar);
        for (int q = 0; q < data.nInterior(); ++q) {

            const bool chemistry_active =
                data.chemistry_block_active[static_cast<size_t>(q)] != 0;
            const bool block_active = chemistry_active;
            data.preconditioner_block_active[static_cast<size_t>(q)] =
                block_active ? 1 : 0;
            if (!block_active) {
                continue;
            }

            MatrixXd block = identity
                - static_cast<double>(gamma)
                    * data.chemistry_jacobians[static_cast<size_t>(q)];

            auto& factor =
                *data.chemistry_factors[static_cast<size_t>(q)];
            if (!factor.compute(block)) {
                data.last_error =
                    "block factorization failed in flow-chemistry preconditioner";
                return 1;
            }
            ++active_blocks;
        }

        data.block_factorization_wall_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - factorization_start).count();
        data.max_active_preconditioner_blocks = std::max(
            data.max_active_preconditioner_blocks, active_blocks);
        *jcur_ptr = (rebuild_jacobian)
            ? SUNTRUE
            : SUNFALSE;
        return 0;
    } catch (const std::exception& error) {
        data.last_error = error.what();
        return 1;
    } catch (...) {
        data.last_error = "unknown block preconditioner setup failure";
        return 1;
    }
}

int chemistryPreconditionerSolve(sunrealtype,
                                 N_Vector,
                                 N_Vector,
                                 N_Vector r,
                                 N_Vector z,
                                 sunrealtype,
                                 sunrealtype,
                                 int,
                                 void* user_data)
{
    auto& data = *static_cast<ImplicitUserData*>(user_data);
    const int nvar = data.input->nVar;
    const sunrealtype* r_values = N_VGetArrayPointer(r);
    sunrealtype* z_values = N_VGetArrayPointer(z);
    if (r_values == nullptr || z_values == nullptr) {
        data.last_error = "preconditioner vector access failed";
        return -1;
    }
    N_VScale(1.0, r, z);

    std::atomic<int> solve_failed{0};
#pragma omp parallel for schedule(static)
    for (int q = 0; q < data.nInterior(); ++q) {
        if (data.preconditioner_block_active[static_cast<size_t>(q)] == 0
            || solve_failed.load(std::memory_order_relaxed) != 0) {
            continue;
        }
        try {
            const sunindextype offset = stateIndex(q, 0, nvar);
            Eigen::Map<const VectorXd> block_rhs(r_values + offset, nvar);
            Eigen::Map<VectorXd> block_solution(z_values + offset, nvar);
            const VectorXd local_rhs = block_rhs;
            block_solution = data.chemistry_factors[
                static_cast<size_t>(q)]->solve(local_rhs);
            if (!block_solution.allFinite()) {
                solve_failed.store(1, std::memory_order_relaxed);
            }
        } catch (...) {
            solve_failed.store(1, std::memory_order_relaxed);
        }
    }
    if (solve_failed.load(std::memory_order_relaxed) != 0) {
        data.last_error = "chemistry preconditioner solve failed";
        return 1;
    }
    return 0;
}

void fillAbsoluteTolerances(N_Vector tolerances,
                            const InputData& input,
                            const ImplicitSolverOptions& options)
{
    sunrealtype* values = N_VGetArrayPointer(tolerances);
    for (int q = 0; q < input.ncell; ++q) {
        values[stateIndex(q, 0, input.nVar)] = options.atol_mass;
        values[stateIndex(q, 1, input.nVar)] = options.atol_momentum;
        values[stateIndex(q, 2, input.nVar)] = options.atol_energy;
        for (int variable = 3; variable < input.nVar; ++variable) {
            values[stateIndex(q, variable, input.nVar)] = options.atol_species;
        }
    }
}

void fillConstraints(N_Vector constraints,
                     const InputData& input,
                     const ImplicitSolverOptions& options)
{
    sunrealtype* values = N_VGetArrayPointer(constraints);
    for (int q = 0; q < input.ncell; ++q) {
        values[stateIndex(q, 0, input.nVar)] = 2.0; // rhoA > 0
        values[stateIndex(q, 1, input.nVar)] = 0.0;
        values[stateIndex(q, 2, input.nVar)] = 0.0;
        for (int variable = 3; variable < input.nVar; ++variable) {
            values[stateIndex(q, variable, input.nVar)] =
                options.enforce_species_positivity ? 1.0 : 0.0;
        }
    }
}

void collectStatistics(void* cvode_memory, ImplicitSolverStats& stats)
{
    CVodeGetNumSteps(cvode_memory, &stats.internal_steps);
    CVodeGetNumRhsEvals(cvode_memory, &stats.rhs_evaluations);
    CVodeGetNumLinRhsEvals(
        cvode_memory, &stats.linear_rhs_evaluations);
    CVodeGetNumJtimesEvals(
        cvode_memory, &stats.jacobian_vector_evaluations);
    CVodeGetNumJTSetupEvals(
        cvode_memory, &stats.jacobian_times_setups);
    CVodeGetNumNonlinSolvIters(cvode_memory, &stats.nonlinear_iterations);
    CVodeGetNumNonlinSolvConvFails(
        cvode_memory, &stats.nonlinear_convergence_failures);
    CVodeGetNumLinIters(cvode_memory, &stats.linear_iterations);
    CVodeGetNumLinConvFails(
        cvode_memory, &stats.linear_convergence_failures);
    CVodeGetNumErrTestFails(cvode_memory, &stats.error_test_failures);
    CVodeGetNumPrecEvals(cvode_memory, &stats.preconditioner_setups);
    CVodeGetNumPrecSolves(cvode_memory, &stats.preconditioner_solves);
    CVodeGetLastStep(cvode_memory, &stats.last_step);
    CVodeGetLastOrder(cvode_memory, &stats.last_order);
}

} // namespace

void prepareFullyImplicitInitialState(MatrixXd& U, const InputData& input)
{
    if (U.rows() != input.nVar || U.cols() != input.ncell) {
        throw std::runtime_error(
            "invalid dimensions for implicit initial-state preparation");
    }
    repairInitialSpeciesRoundoff(U, input, dominantInletSpecies(input));
}

ImplicitSolverStats integrateFullyImplicit(
    MatrixXd& U,
    InputData& input,
    std::shared_ptr<Cantera::Solution> flow_solution,
    const std::vector<double>& output_times,
    const ImplicitSolverOptions& options,
    const ImplicitOutputCallback& output_callback)
{
    ImplicitSolverStats stats;
    validateImplicitOptions(options);
    if (U.rows() != input.nVar || U.cols() != input.ncell || input.ncell < 2) {
        throw std::runtime_error("invalid dimensions for fully implicit state");
    }
    if (output_times.empty()) {
        stats.success = true;
        return stats;
    }
    for (size_t index = 0; index < output_times.size(); ++index) {
        if (!(output_times[index] > 0.0)
            || !std::isfinite(output_times[index])
            || (index > 0 && !(output_times[index] > output_times[index - 1]))) {
            throw std::runtime_error(
                "implicit output times must be positive and strictly increasing");
        }
    }

    ImplicitUserData data(input, std::move(flow_solution), U, options);
    prepareFullyImplicitInitialState(U, input);
    data.U_work = U;
    SUNContext context = nullptr;
    N_Vector y = nullptr;
    N_Vector absolute_tolerances = nullptr;
    N_Vector constraints = nullptr;
    SUNLinearSolver linear_solver = nullptr;
    void* cvode_memory = nullptr;

    auto cleanup = [&]() {
        if (cvode_memory != nullptr) CVodeFree(&cvode_memory);
        if (linear_solver != nullptr) SUNLinSolFree(linear_solver);
        if (constraints != nullptr) N_VDestroy(constraints);
        if (absolute_tolerances != nullptr) N_VDestroy(absolute_tolerances);
        if (y != nullptr) N_VDestroy(y);
        if (context != nullptr) SUNContext_Free(&context);
    };

    try {
        if (SUNContext_Create(SUN_COMM_NULL, &context) != 0) {
            throw std::runtime_error("SUNContext_Create failed");
        }
        y = N_VNew_Serial(data.vectorSize(), context);
        absolute_tolerances = N_VNew_Serial(data.vectorSize(), context);
        constraints = N_VNew_Serial(data.vectorSize(), context);
        if (y == nullptr || absolute_tolerances == nullptr || constraints == nullptr) {
            throw std::runtime_error("failed to allocate SUNDIALS vectors");
        }
        packInterior(U, y, input);
        fillAbsoluteTolerances(absolute_tolerances, input, options);
        fillConstraints(constraints, input, options);

        cvode_memory = CVodeCreate(CV_BDF, context);
        if (cvode_memory == nullptr
            || CVodeInit(cvode_memory, fullyCoupledRhs, 0.0, y) != CV_SUCCESS) {
            throw std::runtime_error("failed to initialize CVODE BDF");
        }
        data.cvode_memory = cvode_memory;
        if (CVodeSetUserData(cvode_memory, &data) != CV_SUCCESS
            || CVodeSVtolerances(cvode_memory, options.rtol,
                                 absolute_tolerances) != CV_SUCCESS
            || CVodeSetConstraints(cvode_memory, constraints) != CV_SUCCESS
            || CVodeSetMaxNumSteps(cvode_memory,
                                   options.max_internal_steps) != CV_SUCCESS
            || CVodeSetMaxOrd(cvode_memory, options.max_order) != CV_SUCCESS) {
            throw std::runtime_error("failed to configure CVODE controls");
        }
        const double effective_max_step = effectiveMaximumStep(options);
        const double effective_initial_step =
            options.initial_step > 0.0 && effective_max_step > 0.0
                ? std::min(options.initial_step, effective_max_step)
                : options.initial_step;
        if (effective_initial_step > 0.0
            && CVodeSetInitStep(cvode_memory, effective_initial_step)
               != CV_SUCCESS) {
            throw std::runtime_error("CVodeSetInitStep failed");
        }
        if (effective_max_step > 0.0
            && CVodeSetMaxStep(cvode_memory, effective_max_step)
               != CV_SUCCESS) {
            throw std::runtime_error("CVodeSetMaxStep failed");
        }

        const bool use_preconditioner =
            options.use_chemistry_preconditioner;
        const int preconditioner_type =
            use_preconditioner ? SUN_PREC_LEFT : SUN_PREC_NONE;
        linear_solver = SUNLinSol_SPGMR(
            y, preconditioner_type, options.krylov_dimension, context);
        if (linear_solver == nullptr
            || CVodeSetLinearSolver(cvode_memory, linear_solver, nullptr)
               != CVLS_SUCCESS) {
            throw std::runtime_error("failed to attach SPGMR to CVODE");
        }

        if (use_preconditioner
            && CVodeSetPreconditioner(
                   cvode_memory, chemistryPreconditionerSetup,
                   chemistryPreconditionerSolve) != CVLS_SUCCESS) {
            throw std::runtime_error(
                "failed to attach block preconditioner to CVODE");
        }

        double reached_time = 0.0;
        for (double target_time : output_times) {
            const int flag = CVode(
                cvode_memory, target_time, y, &reached_time, CV_NORMAL);
            stats.cvode_flag = flag;
            stats.reached_time = reached_time;
            if (flag < 0) {
                collectStatistics(cvode_memory, stats);
                stats.max_active_preconditioner_blocks =
                    data.max_active_preconditioner_blocks;
                stats.chemistry_jacobian_builds =
                    data.chemistry_jacobian_builds;
                stats.chemistry_jacobian_wall_seconds =
                    data.chemistry_jacobian_wall_seconds;
                stats.block_factorization_wall_seconds =
                    data.block_factorization_wall_seconds;
                stats.full_rhs_calls = data.rhs_call_count;
                if (!data.last_error.empty()) {
                    std::cerr << "CVODE RHS detail: " << data.last_error
                              << std::endl;
                }
                cleanup();
                return stats;
            }

            unpackInterior(y, data);
            update_MassFlow_by_time(reached_time, input);
            update_mix_by_time(reached_time, input);
            U = data.U_work;
            if (output_callback) output_callback(reached_time, U);
        }

        collectStatistics(cvode_memory, stats);
        stats.max_active_preconditioner_blocks = data.max_active_preconditioner_blocks;
        stats.chemistry_jacobian_builds =
            data.chemistry_jacobian_builds;
        stats.chemistry_jacobian_wall_seconds =
            data.chemistry_jacobian_wall_seconds;
        stats.block_factorization_wall_seconds =
            data.block_factorization_wall_seconds;
        stats.success = true;
        stats.full_rhs_calls = data.rhs_call_count;
        cleanup();
        return stats;
    } catch (...) {
        cleanup();
        throw;
    }
}
