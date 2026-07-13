#include <cantera/core.h>
#include "cantera/zeroD/ReactorFactory.h"
#include "cantera/zeroD/ReactorNet.h"
#include <Eigen/Dense>
#include <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "preComputation.H"
#include "boundaryCondition.H"
#include "coreMethod.H"


using namespace Eigen;
using namespace Cantera;
using namespace std;



MatrixXd minmod(const MatrixXd & a, const MatrixXd & b)
{
    MatrixXd abs_min = a.array().abs().min(b.array().abs());
    auto sign_match = (a.array() * b.array()) > 0;
    MatrixXd result = a.array().sign()*abs_min.array();
    result = sign_match.select(result, 0.0);
    return result;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
difference_format_NND(const MatrixXd & U_dummy, const int ndummy, const InputData & input)
{
    // NND reconstruction on ghost-padded U_dummy → interface left/right states
    int nVar = input.nVar;
    int nface = input.nface;
    MatrixXd delta_La = U_dummy.block(0, ndummy+1, nVar, nface) - U_dummy.block(0, ndummy, nVar, nface);
    MatrixXd delta_Lb = U_dummy.block(0, ndummy, nVar, nface) - U_dummy.block(0, ndummy-1, nVar, nface);
    MatrixXd U_Lc = U_dummy.block(0, ndummy, nVar, nface) + 0.5 * minmod(delta_La, delta_Lb);

    MatrixXd delta_Ra = U_dummy.block(0, ndummy, nVar, nface) - U_dummy.block(0, ndummy+1, nVar, nface);
    MatrixXd delta_Rb = U_dummy.block(0, ndummy+1, nVar, nface) - U_dummy.block(0, ndummy+2, nVar, nface);
    MatrixXd U_Rc = U_dummy.block(0, ndummy+1, nVar, nface) + 0.5 * minmod(delta_Ra, delta_Rb);
    return {U_Lc, U_Rc};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
baseVar_From_conserVar(const MatrixXd & U)
{
    MatrixXd rhoA = U.row(0);
    MatrixXd rhoA_u = U.row(1);
    MatrixXd u = rhoA_u.array() / rhoA.array();

    MatrixXd rhoA_et = U.row(2);
    MatrixXd et = rhoA_et.array() / rhoA.array();
    MatrixXd e = et.array() - 0.5 * u.array().square();

    MatrixXd rhoA_Y = U.block(3, 0, U.rows() - 3, U.cols());
    MatrixXd Y = (rhoA_Y.array().rowwise() / rhoA.row(0).array()).matrix();

    return {rhoA, u, e, Y};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
PTM_From_baseVar(const MatrixXd & rhoA, const MatrixXd & u, const MatrixXd & e, const MatrixXd & Y, ThermoPhase & gas)
{
    MatrixXd pA(1, Y.cols());
    MatrixXd T(1, Y.cols());
    MatrixXd M(1, Y.cols());

    for (int i = 0; i < Y.cols(); i++)
    {
        gas.setMassFractions(Y.col(i).data());
        gas.setState_UV(e(0, i), 1.0 / rhoA(0, i));
        pA(0, i) = gas.pressure();
        T(0, i) = gas.temperature();    
        M(0, i) = u(0, i) / gas.soundSpeed();
    }
    return {pA, T, M};
}




MatrixXd HLLC_method(const MatrixXd & rho_L, const MatrixXd & u_L, const MatrixXd & e_L, const MatrixXd & Y_L, 
                     const MatrixXd & rho_R, const MatrixXd & u_R, const MatrixXd & e_R, const MatrixXd & Y_R, 
                     const MatrixXd & U_L, const MatrixXd & U_R,
                     const InputData & input, ThermoPhase & gas)
{
    // Thermodynamic closure on left/right reconstructed states
    MatrixXd et_L = e_L.array() + 0.5 * u_L.array().square();
    MatrixXd et_R = e_R.array() + 0.5 * u_R.array().square();

    MatrixXd p_L(1, input.nface), p_R(1, input.nface), a_L(1, input.nface), a_R(1, input.nface);
    MatrixXd gamma_L(1, input.nface), gamma_R(1, input.nface);

    for (int i = 0; i < input.nface; i++)
    {
        gas.setMassFractions(Y_L.col(i).data());
        gas.setState_UV(e_L(0, i), 1.0 / rho_L(0, i));
        p_L(0, i) = gas.pressure();
        a_L(0, i) = gas.soundSpeed();
        gamma_L(0, i) = gas.cp_mass() / gas.cv_mass();

        gas.setMassFractions(Y_R.col(i).data());
        gas.setState_UV(e_R(0, i), 1.0 / rho_R(0, i));
        p_R(0, i) = gas.pressure();
        a_R(0, i) = gas.soundSpeed();
        gamma_R(0, i) = gas.cp_mass() / gas.cv_mass();
    }

    // Physical fluxes F_L / F_R (rho may be rhoA for variable-area form)
    MatrixXd F_L(input.nVar, input.nface), F_R(input.nVar, input.nface);
    F_L << rho_L.array() * u_L.array(),
           rho_L.array() * u_L.array().square() + p_L.array(),
           (rho_L.array() * et_L.array() + p_L.array()) * u_L.array(),
           (rho_L.array() * u_L.array()).replicate(Y_L.rows(), 1) * Y_L.array();
    F_R << rho_R.array() * u_R.array(),
           rho_R.array() * u_R.array().square() + p_R.array(),
           (rho_R.array() * et_R.array() + p_R.array()) * u_R.array(),
           (rho_R.array() * u_R.array()).replicate(Y_R.rows(), 1) * Y_R.array();

    // Roe averages and wave-speed estimates
    MatrixXd sqrt_rho_L = rho_L.array().sqrt();
    MatrixXd sqrt_rho_R = rho_R.array().sqrt();
    MatrixXd denominator = sqrt_rho_L.array() + sqrt_rho_R.array();
    MatrixXd u_Roe = (sqrt_rho_L.array() * u_L.array() + sqrt_rho_R.array() * u_R.array()) / denominator.array();
    MatrixXd a_Roe = (sqrt_rho_L.array() * a_L.array() + sqrt_rho_R.array() * a_R.array()) / denominator.array();
    MatrixXd S_L = (u_L.array() - a_L.array()).min(u_Roe.array() - a_Roe.array());
    MatrixXd S_R = (u_R.array() + a_R.array()).max(u_Roe.array() + a_Roe.array());

    // Contact (star) speed
    MatrixXd numerator = p_R.array() - p_L.array() + 
                         rho_L.array() * u_L.array() * (S_L.array() - u_L.array()) - 
                         rho_R.array() * u_R.array() * (S_R.array() - u_R.array());
    MatrixXd denominator_star = rho_L.array() * (S_L.array() - u_L.array()) - rho_R.array() * (S_R.array() - u_R.array());
    MatrixXd S_star = (numerator.array() / denominator_star.array()).matrix();

    // Star-region conserved states and fluxes
    MatrixXd UL_star(input.nVar, input.nface);
    MatrixXd UR_star(input.nVar, input.nface);

    MatrixXd factor_L = rho_L.array() * (S_L.array() - u_L.array()) / (S_L.array() - S_star.array());
    MatrixXd factor_R = rho_R.array() * (S_R.array() - u_R.array()) / (S_R.array() - S_star.array());
    UL_star << factor_L.array(),
               factor_L.array() * S_star.array(),
               factor_L.array() * (et_L.array() + (S_star.array() - u_L.array()) * (S_star.array() + p_L.array() / rho_L.array()) / (S_L.array() - u_L.array())),
               (factor_L.array().replicate(Y_L.rows(), 1)) * Y_L.array();

    UR_star << factor_R.array(),
               factor_R.array() * S_star.array(),
               factor_R.array() * (et_R.array() + (S_star.array() - u_R.array()) * (S_star.array() + p_R.array() / rho_R.array()) / (S_R.array() - u_R.array())),
               (factor_R.array().replicate(Y_R.rows(), 1)) * Y_R.array();
    MatrixXd FL_star = F_L.array() + (S_L.array().replicate(input.nVar, 1)) * (UL_star.array() - U_L.array());
    MatrixXd FR_star = F_R.array() + (S_R.array().replicate(input.nVar, 1)) * (UR_star.array() - U_R.array());
    // Select flux by (S_L, S_*, S_R)
    MatrixXd Fc(input.nVar, input.nface);
    for (int j = 0; j < input.nface; ++j) {
        double sL = S_L(0, j);
        double sR = S_R(0, j);
        double sStar = S_star(0, j);

        if (sL >= 0.0)
            Fc.col(j) = F_L.col(j);
        else if (sStar >= 0.0)
            Fc.col(j) = FL_star.col(j);
        else if (sR >= 0.0)
            Fc.col(j) = FR_star.col(j);
        else
            Fc.col(j) = F_R.col(j);
    }
    return Fc;
}


MatrixXd dFdx_calculation(MatrixXd & U, ThermoPhase & gas,
                          const InputData & input)
{
    // Pad ghosts, NND reconstruct, HLLC flux (pass rhoA as "rho")
    int ndummy = 3;
    MatrixXd U_dummy = U;
    for (int i = 0; i < ndummy; i++)
    {
        U_dummy = extrapolation(U_dummy);
    }

    auto [U_Lc, U_Rc] = difference_format_NND(U_dummy, ndummy, input);
    auto[rhoA_L, u_L, e_L, Y_L] = baseVar_From_conserVar(U_Lc);
    auto[rhoA_R, u_R, e_R, Y_R] = baseVar_From_conserVar(U_Rc);
    MatrixXd Fc = HLLC_method(rhoA_L, u_L, e_L, Y_L, rhoA_R, u_R, e_R, Y_R, U_Lc, U_Rc, input, gas); 
    const VectorXd & xface = input.xface;
    // Backward difference dF/dx; inlet ghost → dF/dx = 0
    MatrixXd Fc_temp(input.nVar, input.nface + 2);
    Fc_temp.block(0, 1, input.nVar, input.nface) = Fc;
    Fc_temp.col(Fc_temp.cols() - 1) = 2 * Fc.col(Fc.cols() - 1) - Fc.col(Fc.cols() - 2);
    Fc_temp.col(0) = 2 * Fc.col(0) - Fc.col(1);

    MatrixXd xface_temp(1, input.nface + 2);
    xface_temp.block(0, 1, 1, input.nface) = xface.transpose();
    xface_temp(0, xface_temp.cols() - 1) =
        2 * xface(xface.size() - 1) - xface(xface.size() - 2);
    xface_temp(0, 0) = 2 * xface(0) - xface(1);
    const MatrixXd dF = Fc_temp.block(0, 1, input.nVar, input.nface)
                        - Fc_temp.block(0, 0, input.nVar, input.nface);
    const RowVectorXd dx = xface_temp.block(0, 1, 1, input.nface)
                           - xface_temp.block(0, 0, 1, input.nface);
    MatrixXd dFdx = dF.array().rowwise() / dx.array();

    return dFdx;
}


MatrixXd& U_correctY(MatrixXd & U)
{
   // Last species from ΣY_k = 1 (only nSpecies−1 are independently advanced)
   const int nSp = U.rows() - 3;
   MatrixXd Y = (U.block(3, 0, nSp, U.cols()).array().rowwise() / U.row(0).array()).matrix();
   MatrixXd Y_1 = Y.block(0, 0, nSp - 1, Y.cols());
   MatrixXd Y_sum = Y_1.colwise().sum();
   MatrixXd Y_2 = MatrixXd::Constant(1, Y.cols(), 1.0) - Y_sum;
   MatrixXd Y_correct(nSp, Y.cols());
   Y_correct.block(0, 0, nSp - 1, Y.cols()) = Y_1;
   Y_correct.row(nSp - 1) = Y_2;
   U.block(3, 0, nSp, U.cols()) =
       (Y_correct.array().rowwise() * U.row(0).array()).matrix();
   return U;
}

ReactorPool init_reactor_pool(std::shared_ptr<Cantera::Solution> sol, int n_threads)
{
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    ReactorPool pool;
    pool.sol = sol;
    pool.n_threads = n_threads;
    pool.reactors.reserve(n_threads);
    pool.nets.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        pool.reactors.push_back(
            Cantera::newReactor4("IdealGasReactor", sol, true));
        pool.nets.push_back(
            std::make_unique<Cantera::ReactorNet>(pool.reactors.back()));
    }
    return pool;
}


MatrixXd& reactor_calculation(MatrixXd& U, ReactorPool& pool, const InputData& input)
{
    MatrixXd rhoA = U.row(0);
    MatrixXd rhou = U.row(1);
    MatrixXd u = rhou.array() / rhoA.array();
    MatrixXd rho_et = U.row(2);
    MatrixXd et = rho_et.array() / rhoA.array();
    MatrixXd e = et.array() - 0.5 * u.array().square();
    MatrixXd rhoA_Y = U.block(3, 0, U.rows() - 3, U.cols());
    MatrixXd rho = rhoA.array() / input.A_array.array();
    MatrixXd Y = (rhoA_Y.array().rowwise() / rhoA.row(0).array()).matrix();

#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < input.nface; i++)
    {
        const int j = omp_get_thread_num();
        auto& gas = *pool.reactors[j]->phase()->thermo();
        auto& reactor = *pool.reactors[j];
        auto& net = *pool.nets[j];

        gas.setMassFractions(Y.col(i).data());
        gas.setState_UV(e(0, i), 1.0 / rho(0, i));
        reactor.syncState();
        net.setInitialTime(0.0);
        net.advance(input.dt);

        gas.getMassFractions(Y.col(i).data());
    }

    U.block(3, 0, U.rows() - 3, U.cols()) =
        (Y.array().rowwise() * rhoA.row(0).array()).matrix();
    return U;
}