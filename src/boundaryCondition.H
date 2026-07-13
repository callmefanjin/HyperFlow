#ifndef BOUNDARY_CONDITION_H
#define BOUNDARY_CONDITION_H

#include <Eigen/Dense>

#include "preComputation.H"

namespace Cantera
{
class ThermoPhase;
}

/** Extract rho, u, p, Y, Ma from a conserved column U and local area */
void rho_u_p_Y_M_extract(const Eigen::MatrixXd & U,
                         double local_A,
                         Cantera::ThermoPhase & gas,
                         double & rho,
                         double & u,
                         double & p,
                         double & Ma,
                         Eigen::MatrixXd & Y);

/**
 * Outlet conserved column (nVar×1).
 * b = boundary, d = interior left of exit, a = ambient (p_a, c_a, rho_a).
 * Interior Ma ≥ 1: zero gradient (copy U.col(N-2));
 * subsonic: prescribe p_a with linearized Riemann + Cantera closure.
 */
Eigen::MatrixXd generateOutletBoundaryVector(const Eigen::MatrixXd & U,
                                             double A_d,
                                             double A_b,
                                             double p_a,
                                             double c_a,
                                             double rho_a,
                                             Cantera::ThermoPhase & gas);

/**
 * Inlet conserved column (nVar×1).
 * a = exterior reservoir, b = boundary, d = interior right of inlet.
 * Subsonic: reservoir–interior Riemann; supersonic: fully prescribed reservoir.
 */
Eigen::MatrixXd generateInletBoundaryVector(const Eigen::MatrixXd & U,
                                            double A_b,
                                            double A_d,
                                            double p_a,
                                            double c_a,
                                            double rho_a,
                                            double u_a,
                                            const Eigen::MatrixXd & Y_a,
                                            Cantera::ThermoPhase & gas);

/** Overwrite first (inlet) and last (outlet) columns of U */
void boundaryUpdate(Eigen::MatrixXd & U,
                    const InputData & input,
                    Cantera::ThermoPhase & gas);

/**
 * Add one ghost column on each side: 2*U_b − U_d.
 * Call after boundaryUpdate so col(0)/col(N-1) hold BC states.
 */
Eigen::MatrixXd extrapolation(const Eigen::MatrixXd & U);

#endif
