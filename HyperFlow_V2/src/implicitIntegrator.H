#ifndef IMPLICIT_INTEGRATOR_H
#define IMPLICIT_INTEGRATOR_H

#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "preComputation.H"

namespace Cantera
{
class Solution;
}

struct IgnitionSourceOptions
{
    bool enabled = false;
    double total_energy = 0.0;
    double center = 0.0;
    double half_length = 0.0;
    double start_time = 0.0;
    double duration = 0.0;

    // Derived from the constant-area geometry and prescribed total energy.
    double area = 0.0;
    double peak_volumetric_power = 0.0;
};

struct ImplicitSolverOptions
{
    double rtol = 1.0e-6;
    double atol_mass = 1.0e-12;
    double atol_momentum = 1.0e-8;
    double atol_energy = 1.0e-3;
    double atol_species = 1.0e-14;
    double initial_step = 0.0;
    double max_step = 0.0;
    long max_internal_steps = 1000000;
    int max_order = 2;
    int krylov_dimension = 30;
    bool use_chemistry_preconditioner = false;
    bool enforce_species_positivity = true;
    IgnitionSourceOptions ignition_source;
};

struct ImplicitSolverStats
{
    bool success = false;
    int cvode_flag = 0;
    double reached_time = 0.0;
    long internal_steps = 0;
    long rhs_evaluations = 0;
    long nonlinear_iterations = 0;
    long linear_rhs_evaluations = 0;
    long jacobian_vector_evaluations = 0;
    long jacobian_times_setups = 0;
    long full_rhs_calls = 0;
    long nonlinear_convergence_failures = 0;
    long linear_iterations = 0;
    long linear_convergence_failures = 0;
    long error_test_failures = 0;
    long preconditioner_setups = 0;
    long preconditioner_solves = 0;
    long max_active_preconditioner_blocks = 0;
    double last_step = 0.0;
    int last_order = 0;
    long chemistry_jacobian_builds = 0;
    double chemistry_jacobian_wall_seconds = 0.0;
    double block_factorization_wall_seconds = 0.0;
};

using ImplicitOutputCallback =
    std::function<void(double, const Eigen::MatrixXd&)>;

// Validate and repair only roundoff-scale initial species errors using the
// same dependent bath-gas identity as the implicit RHS. Materially invalid
// states are rejected instead of being projected silently.
void prepareFullyImplicitInitialState(Eigen::MatrixXd& U,
                                      const InputData& input);

// Fully coupled method-of-lines integration with CVODE BDF. The state contains
// interior conservative variables. Boundary values are reconstructed for each
// RHS evaluation; flow and chemistry are evaluated together without splitting.
ImplicitSolverStats integrateFullyImplicit(
    Eigen::MatrixXd& U,
    InputData& input,
    std::shared_ptr<Cantera::Solution> flow_solution,
    const std::vector<double>& output_times,
    const ImplicitSolverOptions& options,
    const ImplicitOutputCallback& output_callback = {});

#endif

// end of implicit integrator interface

