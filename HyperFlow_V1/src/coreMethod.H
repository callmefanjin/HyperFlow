#ifndef CORE_METHOD_H
#define CORE_METHOD_H

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <vector>
#include "cantera/zeroD/Reactor.h"
#include "cantera/zeroD/ReactorNet.h"

#include "preComputation.H"

namespace Cantera
{
class ThermoPhase;
class Solution;
}

/** NND minmod limiter */
Eigen::MatrixXd minmod(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b);

/** NND reconstruction: left/right interface states from ghost-padded U_dummy */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
difference_format_NND(const Eigen::MatrixXd & U_dummy, int ndummy, const InputData & input);

/** Conserved U → (rhoA, u, e, Y) */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
baseVar_From_conserVar(const Eigen::MatrixXd & U);

/** Primitive closure (rhoA, u, e, Y) → (pA, T, M) via Cantera */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
PTM_From_baseVar(const Eigen::MatrixXd & rhoA,
                 const Eigen::MatrixXd & u,
                 const Eigen::MatrixXd & e,
                 const Eigen::MatrixXd & Y,
                 Cantera::ThermoPhase & gas);

/** HLLC numerical flux; callers may pass rhoA as the "rho" arguments */
Eigen::MatrixXd HLLC_method(const Eigen::MatrixXd & rho_L,
                            const Eigen::MatrixXd & u_L,
                            const Eigen::MatrixXd & e_L,
                            const Eigen::MatrixXd & Y_L,
                            const Eigen::MatrixXd & rho_R,
                            const Eigen::MatrixXd & u_R,
                            const Eigen::MatrixXd & e_R,
                            const Eigen::MatrixXd & Y_R,
                            const Eigen::MatrixXd & U_L,
                            const Eigen::MatrixXd & U_R,
                            const InputData & input,
                            Cantera::ThermoPhase & gas);

/** Flux divergence dF/dx (backward difference; inlet dF/dx = 0 via ghost) */
Eigen::MatrixXd dFdx_calculation(Eigen::MatrixXd & U,
                                   Cantera::ThermoPhase & gas,
                                   const InputData & input);

/** Enforce ΣY_k = 1 on the last species (updates ρA Y_k in place) */
Eigen::MatrixXd & U_correctY(Eigen::MatrixXd & U);

/** One IdealGasReactor + ReactorNet per OpenMP thread */
struct ReactorPool {
    std::shared_ptr<Cantera::Solution> sol;
    std::vector<std::shared_ptr<Cantera::ReactorBase>> reactors;
    std::vector<std::unique_ptr<Cantera::ReactorNet>> nets;
    int n_threads = 1;
};

/** Build reactor pool; n_threads <= 0 → omp_get_max_threads() */
ReactorPool init_reactor_pool(std::shared_ptr<Cantera::Solution> sol,
                              int n_threads = 0);

/** Chemistry substep: constant-volume IdealGasReactor for dt at each face */
Eigen::MatrixXd & reactor_calculation(Eigen::MatrixXd & U,
                                        ReactorPool& pool,
                                        const InputData & input);

#endif
