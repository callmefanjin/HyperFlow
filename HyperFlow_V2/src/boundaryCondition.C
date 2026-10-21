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
    // 考虑超声速出口和亚声速出口
    // b标识边界点，d标识出口左侧的内点，a标识出口右侧的外部环境
    MatrixXd U_d = U.col(U.cols() - 1);
    double rho_d, u_d, p_d, Ma_d;
    MatrixXd Y_d;
    rho_u_p_Y_M_extract(U_d, A_d, gas, rho_d, u_d, p_d, Ma_d, Y_d);
    if (Ma_d >= 1.0) {
        if (!(A_d > 0.0) || !(A_b > 0.0)) {
            throw std::runtime_error("non-positive area at supersonic outlet");
        }
        // Extrapolate rho, u, e and Y while preserving the area-integrated
        // conservative-state design: U_b = A_b * (U_d / A_d).
        return U_d * (A_b / A_d);
    } else {
        // At a subsonic outflow, the outgoing acoustic characteristic is
        // carried by the interior state. Therefore the pressure correction
        // must use the interior acoustic impedance rho_d*c_d, rather than the
        // far-field/environment impedance. The latter can be orders of
        // magnitude smaller and creates an almost-vacuum, extremely fast ghost
        // state when chamber pressure is appreciably above the back pressure.
        const double c_d = gas.soundSpeed();
        if (!(rho_d > 0.0) || !(c_d > 0.0)) {
            throw std::runtime_error(
                "invalid interior state at subsonic outlet");
        }
        double p_b = p_a;
        double rho_b = rho_d + (p_b - p_d) / (c_d * c_d);
        double u_b = u_d + (p_d - p_b) / (rho_d * c_d);
        if (!(rho_b > 0.0)) {
            throw std::runtime_error(
                "non-positive characteristic outlet density");
        }
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
    // a表示入口左侧的环境，b表示入口边界点，d表示入口右侧的内点
    // 判断入口超声速还是亚声速
    MatrixXd U_d = U.col(0);
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

MatrixXd addBoundaryGhostCells(const MatrixXd& U,
                                      const InputData& input,
                                      ThermoPhase& gas)
{
    if (U.cols() != input.ncell || input.ncell < 2) {
        throw std::runtime_error("boundary ghosts require ncell physical states");
    }
    const double A_d_in = input.A_array(0, 0);
    const double A_d_out = input.A_array(0, input.ncell - 1);
    const double A_g_in = 2.0 * input.A_face_array(0, 0) - A_d_in;
    const double A_g_out = 2.0 * input.A_face_array(0, input.nface - 1) - A_d_out;
    if (!(A_g_in > 0.0) || !(A_g_out > 0.0)) {
        throw std::runtime_error("non-positive mirrored ghost-cell area");
    }

    gas.setState_TPX(input.temperature_environment, input.pressure_environment,
                     input.airMoleFraction);
    const MatrixXd U_outlet = generateOutletBoundaryVector(
        U, A_d_out, A_g_out, input.pressure_environment, gas.soundSpeed(),
        gas.density(), gas);

    gas.setState_TPX(input.temperature_inlet, input.pressure_inlet,
                     input.airMoleFraction);
    MatrixXd Y_inlet = MatrixXd::Zero(gas.nSpecies(), 1);
    gas.getMassFractions(Y_inlet.data());
    const MatrixXd U_inlet = generateInletBoundaryVector(
        U, A_g_in, A_d_in, input.pressure_inlet, gas.soundSpeed(),
        gas.density(), input.velocity_inlet, Y_inlet, gas);

    MatrixXd result(U.rows(), U.cols() + 2);
    result.col(0) = U_inlet;
    result.middleCols(1, U.cols()) = U;
    result.col(result.cols() - 1) = U_outlet;
    return result;
}
MatrixXd extrapolation(const MatrixXd & U)
{
    MatrixXd U_outlet = 2*U.col(U.cols() - 1) - U.col(U.cols() - 2);
    MatrixXd U_inlet = 2*U.col(0) - U.col(1);
    MatrixXd U_extrapolation = MatrixXd::Zero(U.rows(), U.cols() + 2);
    U_extrapolation.block(0, 1, U.rows(), U.cols()) = U;
    U_extrapolation.col(0) = U_inlet;
    U_extrapolation.col(U_extrapolation.cols() - 1) = U_outlet;
    return U_extrapolation;
}
