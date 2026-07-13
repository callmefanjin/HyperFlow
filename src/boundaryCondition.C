#include "boundaryCondition.H"

#include "cantera/core.h"

#include <stdexcept>
#include <string>

using namespace Eigen;
using namespace Cantera;

void rho_u_p_Y_M_extract(const MatrixXd & U,
                         const double local_A,
                         ThermoPhase & gas,
                         double & rho,
                         double & u,
                         double & p,
                         double & Ma,
                         MatrixXd & Y)
{
    double rhoA = U(0, 0);
    double rhoA_u = U(1, 0);
    double rhoA_et = U(2, 0);
    MatrixXd rhoA_Y = U.block(3, 0, U.rows() - 3, U.cols());
    rho = rhoA / local_A;
    u = rhoA_u / rhoA;
    double e = rhoA_et / rhoA - 0.5 * u * u;
    Y = rhoA_Y.array() / rhoA;
    gas.setMassFractions(Y.col(0).data());
    gas.setState_UV(e, 1.0 / rho);
    p = gas.pressure();
    Ma = abs(u) / gas.soundSpeed();
}

MatrixXd generateOutletBoundaryVector(const MatrixXd & U,
                                      const double A_d,
                                      const double A_b,
                                      const double p_a,
                                      const double c_a,
                                      const double rho_a,
                                      ThermoPhase & gas)
{
    // Outlet: b = boundary, d = interior, a = ambient
    MatrixXd U_d = U.col(U.cols() - 2);
    double rho_d, u_d, p_d, Ma_d;
    MatrixXd Y_d;
    rho_u_p_Y_M_extract(U_d, A_d, gas, rho_d, u_d, p_d, Ma_d, Y_d);
    if (Ma_d >= 1.0) {
        return U_d;
    } else {
        double p_b = p_a;
        double rho_b = rho_d + (p_b - p_d) / (c_a * c_a); 
        double u_b = u_d + (p_d - p_b) / (rho_a * c_a);
        gas.setMassFractions(Y_d.col(0).data());
        double R = Cantera::GasConstant / gas.meanMolecularWeight();
        gas.setState_TP(p_b/R/rho_b, p_b);
        double e_b = gas.intEnergy_mass();
        double et_b = e_b + 0.5 * u_b * u_b;
        double rhoA_b = rho_b * A_b;
        double rhoA_u_b = rhoA_b * u_b;
        double rhoA_et_b = rhoA_b * et_b;
        MatrixXd rhoA_Y_b = Y_d.array() * rhoA_b;
        MatrixXd U_b = MatrixXd::Zero(U.rows(), 1);
        U_b(0, 0) = rhoA_b;
        U_b(1, 0) = rhoA_u_b;
        U_b(2, 0) = rhoA_et_b;
        U_b.block(3, 0, U_b.rows() - 3, U_b.cols()) = rhoA_Y_b;
        return U_b;
    }
}

MatrixXd generateInletBoundaryVector(const MatrixXd & U,
                                    const double A_b,
                                    const double A_d,
                                    const double p_a,
                                    const double c_a,
                                    const double rho_a,
                                    const double u_a,
                                    const MatrixXd & Y_a,
                                    ThermoPhase & gas)
{
    // Inlet: a = reservoir, b = boundary, d = interior (subsonic vs supersonic)
    MatrixXd U_d = U.col(1);
    double rho_d, u_d, p_d, Ma_d;
    MatrixXd Y_d;
    rho_u_p_Y_M_extract(U_d, A_d, gas, rho_d, u_d, p_d, Ma_d, Y_d);
    if (abs(u_a)/c_a < 1.0) {
        double p_b = 0.5 * (p_a + p_d - rho_a * c_a * (u_a - u_d));
        double rho_b = rho_a + (p_b - p_a) / c_a / c_a;
        double u_b = u_a - (p_a - p_b) / (rho_a * c_a);
        gas.setMassFractions(Y_a.col(0).data());
        double R = Cantera::GasConstant / gas.meanMolecularWeight();
        gas.setState_TP(p_b/R/rho_b, p_b);
        double e_b = gas.intEnergy_mass();
        double et_b = e_b + 0.5 * u_b * u_b;
        double rhoA_b = rho_b * A_b;
        double rhoA_u_b = rhoA_b * u_b;
        double rhoA_et_b = rhoA_b * et_b;
        MatrixXd rhoA_Y_b = Y_a.array() * rhoA_b;
        MatrixXd U_b = MatrixXd::Zero(U.rows(), 1);
        U_b(0, 0) = rhoA_b;
        U_b(1, 0) = rhoA_u_b;
        U_b(2, 0) = rhoA_et_b;
        U_b.block(3, 0, U_b.rows() - 3, U_b.cols()) = rhoA_Y_b;
        return U_b;
    }   
    else {
        double p_b = p_a;
        double rho_b = rho_a;
        double u_b = u_a;
        gas.setMassFractions(Y_a.col(0).data());
        double R = Cantera::GasConstant / gas.meanMolecularWeight();
        gas.setState_TP(p_b/R/rho_b, p_b);
        double e_b = gas.intEnergy_mass();
        double et_b = e_b + 0.5 * u_b * u_b;
        double rhoA_b = rho_b * A_b;
        double rhoA_u_b = rhoA_b * u_b;
        double rhoA_et_b = rhoA_b * et_b;
        MatrixXd rhoA_Y_b = Y_a.array() * rhoA_b;
        MatrixXd U_b = MatrixXd::Zero(U.rows(), 1);
        U_b(0, 0) = rhoA_b;
        U_b(1, 0) = rhoA_u_b;
        U_b(2, 0) = rhoA_et_b;
        U_b.block(3, 0, U_b.rows() - 3, U_b.cols()) = rhoA_Y_b;
        return U_b;
    }
}

void boundaryUpdate(MatrixXd & U,const InputData & input,ThermoPhase & gas)
{
    double A_d_out = input.A_array(0, input.nface - 2);
    double A_b_out = input.A_array(0, input.nface - 1);
    double p_a_out = input.pressure_environment;
    double T_a_out = input.temperature_environment;
    gas.setState_TPX(T_a_out, p_a_out, input.outletMoleFraction);
    double c_a_out = gas.soundSpeed();
    double rho_a_out = gas.density();
    MatrixXd U_outlet = generateOutletBoundaryVector(U, A_d_out, A_b_out, p_a_out, c_a_out, rho_a_out, gas);

    double A_b_in = input.A_array(0, 0);
    double A_d_in = input.A_array(0, 1);
    double p_a_in = input.pressure_inlet;
    double T_a_in = input.temperature_inlet;
    gas.setState_TPX(T_a_in, p_a_in, input.inletMoleFraction);
    MatrixXd Y_a_in = MatrixXd::Zero(gas.nSpecies(), 1);
    gas.getMassFractions(Y_a_in.data());
    double c_a_in = gas.soundSpeed();
    double rho_a_in = gas.density();
    double u_a_in = input.velocity_inlet;
    MatrixXd U_inlet = generateInletBoundaryVector(U, A_b_in, A_d_in, p_a_in, c_a_in, rho_a_in, u_a_in, Y_a_in, gas);
    U.block(0, 0, U.rows(), 1) = U_inlet;
    U.block(0, U.cols() - 1, U.rows(), 1) = U_outlet;
}
MatrixXd extrapolation(const MatrixXd & U)
{
    // Linear extrapolation ghosts at both ends
    MatrixXd U_outlet = 2*U.col(U.cols() - 1) - U.col(U.cols() - 2);
    MatrixXd U_inlet = 2*U.col(0) - U.col(1);
    MatrixXd U_extrapolation = MatrixXd::Zero(U.rows(), U.cols() + 2);
    U_extrapolation.block(0, 1, U.rows(), U.cols()) = U;
    U_extrapolation.col(0) = U_inlet;
    U_extrapolation.col(U_extrapolation.cols() - 1) = U_outlet;
    return U_extrapolation;
}