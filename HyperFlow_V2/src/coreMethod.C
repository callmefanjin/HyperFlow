#include <cantera/core.h>
#include <Eigen/Dense>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <sstream>
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
    // 接受虚网格处理过后的U_dummy, 返回界面左侧和右侧处重构的的U
    //采用NND格式
    int nVar = input.nVar;
    const int ninterface = input.nface;
    MatrixXd delta_La = U_dummy.block(0, ndummy+1, nVar, ninterface) - U_dummy.block(0, ndummy, nVar, ninterface);
    MatrixXd delta_Lb = U_dummy.block(0, ndummy, nVar, ninterface) - U_dummy.block(0, ndummy-1, nVar, ninterface);
    MatrixXd U_Lc = U_dummy.block(0, ndummy, nVar, ninterface) + 0.5 * minmod(delta_La, delta_Lb);

    MatrixXd delta_Ra = U_dummy.block(0, ndummy, nVar, ninterface) - U_dummy.block(0, ndummy+1, nVar, ninterface);
    MatrixXd delta_Rb = U_dummy.block(0, ndummy+1, nVar, ninterface) - U_dummy.block(0, ndummy+2, nVar, ninterface);
    MatrixXd U_Rc = U_dummy.block(0, ndummy+1, nVar, ninterface) + 0.5 * minmod(delta_Ra, delta_Rb);
    return {U_Lc, U_Rc};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
difference_format_first_order(const MatrixXd & U_dummy, const int ndummy,
                              const InputData & input)
{
    const int ninterface = input.nface;
    MatrixXd U_Lc = U_dummy.block(
        0, ndummy, input.nVar, ninterface);
    MatrixXd U_Rc = U_dummy.block(
        0, ndummy + 1, input.nVar, ninterface);
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
    // 从U_L和U_R中提取基本变量
    MatrixXd et_L = e_L.array() + 0.5 * u_L.array().square();
    MatrixXd et_R = e_R.array() + 0.5 * u_R.array().square();

    const int ninterface = static_cast<int>(U_L.cols());
    MatrixXd p_L(1, ninterface), p_R(1, ninterface), a_L(1, ninterface), a_R(1, ninterface);

    for (int i = 0; i < ninterface; i++)
    {
        auto set_face_state = [&](const MatrixXd& rho, const MatrixXd& e,
                                  const MatrixXd& Y, const char* side,
                                  const MatrixXd& u, MatrixXd& p,
                                  MatrixXd& a) {
            try {

                gas.setMassFractions(Y.col(i).data());
                gas.setState_UV(e(0, i), 1.0 / rho(0, i));
                p(0, i) = gas.pressure();
                a(0, i) = gas.soundSpeed();

            } catch (const std::exception& error) {
                std::ostringstream detail;
                detail << std::setprecision(17)
                       << "HLLC " << side << " face state failure"
                       << ", face=" << i
                       << ", rhoA=" << rho(0, i)
                       << ", e=" << e(0, i)
                       << ", u=" << u(0, i)
                       << ", Y_min=" << Y.col(i).minCoeff()
                       << ", Y_max=" << Y.col(i).maxCoeff()
                       << ", Y_sum=" << Y.col(i).sum()
                       << ", reason=" << error.what();
                throw std::runtime_error(detail.str());
            }
        };
        set_face_state(rho_L, e_L, Y_L, "left", u_L, p_L, a_L);
        set_face_state(rho_R, e_R, Y_R, "right", u_R, p_R, a_R);
    }

    // 计算左右的F通量
    MatrixXd F_L(input.nVar, ninterface), F_R(input.nVar, ninterface);
    F_L << rho_L.array() * u_L.array(),  // rho * u
           rho_L.array() * u_L.array().square() + p_L.array(),  // rho * u^2 + p
           (rho_L.array() * et_L.array() + p_L.array()) * u_L.array(),  // (rho * et + p) * u
           (rho_L.array() * u_L.array()).replicate(Y_L.rows(), 1) * Y_L.array();  // rho * u * Y
    F_R << rho_R.array() * u_R.array(),  // rho * u
           rho_R.array() * u_R.array().square() + p_R.array(),  // rho * u^2 + p
           (rho_R.array() * et_R.array() + p_R.array()) * u_R.array(),  // (rho * et + p) * u
           (rho_R.array() * u_R.array()).replicate(Y_R.rows(), 1) * Y_R.array();  // rho * u * Y

    // 计算Roe平均参数
    MatrixXd sqrt_rho_L = rho_L.array().sqrt();
    MatrixXd sqrt_rho_R = rho_R.array().sqrt();
    MatrixXd denominator = sqrt_rho_L.array() + sqrt_rho_R.array();
    MatrixXd u_Roe = (sqrt_rho_L.array() * u_L.array() + sqrt_rho_R.array() * u_R.array()) / denominator.array();
    MatrixXd a_Roe = (sqrt_rho_L.array() * a_L.array() + sqrt_rho_R.array() * a_R.array()) / denominator.array();
    // 估计波速
    MatrixXd S_L = (u_L.array() - a_L.array()).min(u_Roe.array() - a_Roe.array());
    MatrixXd S_R = (u_R.array() + a_R.array()).max(u_Roe.array() + a_Roe.array());

    // 计算中间波速
    MatrixXd numerator = p_R.array() - p_L.array() +
                         rho_L.array() * u_L.array() * (S_L.array() - u_L.array()) -
                         rho_R.array() * u_R.array() * (S_R.array() - u_R.array());
    MatrixXd denominator_star = rho_L.array() * (S_L.array() - u_L.array()) - rho_R.array() * (S_R.array() - u_R.array());
    MatrixXd S_star = (numerator.array() / denominator_star.array()).matrix();

    // 计算中间状态守恒变量
    MatrixXd UL_star(input.nVar, ninterface);
    MatrixXd UR_star(input.nVar, ninterface);

    MatrixXd factor_L = rho_L.array() * (S_L.array() - u_L.array()) / (S_L.array() - S_star.array());
    MatrixXd factor_R = rho_R.array() * (S_R.array() - u_R.array()) / (S_R.array() - S_star.array());
    UL_star << factor_L.array(),
               factor_L.array() * S_star.array(),
               factor_L.array() * (et_L.array() + (S_star.array() - u_L.array()) * (S_star.array() + p_L.array() / (rho_L.array() * (S_L.array() - u_L.array())))),
               (factor_L.array().replicate(Y_L.rows(), 1)) * Y_L.array();

    UR_star << factor_R.array(),
               factor_R.array() * S_star.array(),
               factor_R.array() * (et_R.array() + (S_star.array() - u_R.array()) * (S_star.array() + p_R.array() / (rho_R.array() * (S_R.array() - u_R.array())))),
               (factor_R.array().replicate(Y_R.rows(), 1)) * Y_R.array();
    // 计算中间通量
    MatrixXd FL_star = F_L.array() + (S_L.array().replicate(input.nVar, 1)) * (UL_star.array() - U_L.array());
    MatrixXd FR_star = F_R.array() + (S_R.array().replicate(input.nVar, 1)) * (UR_star.array() - U_R.array());
    // 根据波速条件选择通量
    MatrixXd Fc(input.nVar, ninterface);
    for (int j = 0; j < ninterface; ++j) {
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

MatrixXd dFdx_calculation(const MatrixXd & U, ThermoPhase & gas,
                          const InputData & input,
                          const int spatial_order)
{
    // U stores area-integrated conserved variables. Reconstructing U directly
    // would misinterpret the prescribed area gradient as a fluid-state
    // gradient. Normalize the physical cells and boundary ghosts by their
    // cell-center areas, reconstruct U/A, then multiply both sides by the same
    // interface area. This is the quasi-1D FVM face-state definition.
    const int ndummy = 3;
    const double A_ghost_in =
        2.0 * input.A_face_array(0, 0) - input.A_array(0, 0);
    const double A_ghost_out =
        2.0 * input.A_face_array(0, input.nface - 1)
        - input.A_array(0, input.ncell - 1);
    if (!(A_ghost_in > 0.0) || !(A_ghost_out > 0.0)) {
        throw std::runtime_error(
            "non-positive mirrored ghost-cell area in flux reconstruction");
    }
    const MatrixXd U_with_boundary = addBoundaryGhostCells(U, input, gas);
    MatrixXd A_with_boundary(1, input.ncell + 2);
    A_with_boundary(0, 0) = A_ghost_in;
    A_with_boundary.middleCols(1, input.ncell) = input.A_array;
    A_with_boundary(0, input.ncell + 1) = A_ghost_out;
    MatrixXd q_dummy =
        (U_with_boundary.array().rowwise()
         / A_with_boundary.row(0).array()).matrix();
    for (int i = 0; i < ndummy; ++i) {
        q_dummy = extrapolation(q_dummy);
    }

    MatrixXd U_Lc;
    MatrixXd U_Rc;
    MatrixXd rhoA_L, u_L, e_L, Y_L;
    MatrixXd rhoA_R, u_R, e_R, Y_R;

    if (input.reconstruction_variables == "conservative_per_area") {
        MatrixXd q_Lc;
        MatrixXd q_Rc;
        if (spatial_order == 1) {
            std::tie(q_Lc, q_Rc) =
                difference_format_first_order(q_dummy, ndummy, input);
        } else if (spatial_order == 2) {
            std::tie(q_Lc, q_Rc) =
                difference_format_NND(q_dummy, ndummy, input);
        } else {
            throw std::runtime_error(
                "spatial_order must be 1 (piecewise constant) or 2 (NND)");
        }
        U_Lc =
            (q_Lc.array().rowwise()
             * input.A_face_array.row(0).array()).matrix();
        U_Rc =
            (q_Rc.array().rowwise()
             * input.A_face_array.row(0).array()).matrix();
        std::tie(rhoA_L, u_L, e_L, Y_L) = baseVar_From_conserVar(U_Lc);
        std::tie(rhoA_R, u_R, e_R, Y_R) = baseVar_From_conserVar(U_Rc);
    } else if (input.reconstruction_variables == "primitive") {
        auto [rho_dummy, u_dummy, e_dummy, Y_dummy] =
            baseVar_From_conserVar(q_dummy);
        MatrixXd W_dummy(input.nVar, q_dummy.cols());
        W_dummy.row(0) = rho_dummy;
        W_dummy.row(1) = u_dummy;
        W_dummy.row(2) = e_dummy;
        W_dummy.block(3, 0, input.nSpecies, q_dummy.cols()) = Y_dummy;

        MatrixXd W_Lc;
        MatrixXd W_Rc;
        if (spatial_order == 1) {
            std::tie(W_Lc, W_Rc) =
                difference_format_first_order(W_dummy, ndummy, input);
        } else if (spatial_order == 2) {
            std::tie(W_Lc, W_Rc) =
                difference_format_NND(W_dummy, ndummy, input);
        } else {
            throw std::runtime_error(
                "spatial_order must be 1 (piecewise constant) or 2 (NND)");
        }
        rhoA_L =
            (W_Lc.row(0).array()
             * input.A_face_array.row(0).array()).matrix();
        rhoA_R =
            (W_Rc.row(0).array()
             * input.A_face_array.row(0).array()).matrix();
        u_L = W_Lc.row(1);
        u_R = W_Rc.row(1);
        e_L = W_Lc.row(2);
        e_R = W_Rc.row(2);
        Y_L = W_Lc.block(3, 0, input.nSpecies, input.nface);
        Y_R = W_Rc.block(3, 0, input.nSpecies, input.nface);

        auto normalizeFaceMassFractions = [](MatrixXd & Y_face,
                                              const char * side) {
            for (int face = 0; face < Y_face.cols(); ++face) {
                double sum = 0.0;
                for (int species = 0; species < Y_face.rows(); ++species) {
                    double & value = Y_face(species, face);
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                            std::string("non-finite primitive face mass fraction on ")
                            + side);
                    }
                    value = std::max(0.0, value);
                    sum += value;
                }
                if (!(sum > 0.0) || !std::isfinite(sum)) {
                    throw std::runtime_error(
                        std::string("invalid primitive face composition on ")
                        + side);
                }
                Y_face.col(face) /= sum;
            }
        };
        normalizeFaceMassFractions(Y_L, "left");
        normalizeFaceMassFractions(Y_R, "right");

        if (!rhoA_L.allFinite() || !rhoA_R.allFinite()
            || (rhoA_L.array() <= 0.0).any() || (rhoA_R.array() <= 0.0).any()
            || !u_L.allFinite() || !u_R.allFinite()
            || !e_L.allFinite() || !e_R.allFinite()) {
            throw std::runtime_error("invalid primitive reconstructed face state");
        }

        U_Lc.resize(input.nVar, input.nface);
        U_Rc.resize(input.nVar, input.nface);
        U_Lc.row(0) = rhoA_L;
        U_Rc.row(0) = rhoA_R;
        U_Lc.row(1) = (rhoA_L.array() * u_L.array()).matrix();
        U_Rc.row(1) = (rhoA_R.array() * u_R.array()).matrix();
        U_Lc.row(2) =
            (rhoA_L.array()
             * (e_L.array() + 0.5 * u_L.array().square())).matrix();
        U_Rc.row(2) =
            (rhoA_R.array()
             * (e_R.array() + 0.5 * u_R.array().square())).matrix();
        U_Lc.block(3, 0, input.nSpecies, input.nface) =
            (Y_L.array().rowwise() * rhoA_L.row(0).array()).matrix();
        U_Rc.block(3, 0, input.nSpecies, input.nface) =
            (Y_R.array().rowwise() * rhoA_R.row(0).array()).matrix();
    } else {
        throw std::runtime_error("unknown reconstruction_variables option");
    }
    // 应用HLLC方法
    // 这里是故意将rhoA当作rho传入，并非错误，而是为了方便计算，相当于对一个管道截面积积分的结果作为rho传入
    MatrixXd Fc = HLLC_method(
        rhoA_L, u_L, e_L, Y_L, rhoA_R, u_R, e_R, Y_R,
        U_Lc, U_Rc, input, gas);
    // FVM: one shared numerical flux per interface; internal fluxes cancel.
    MatrixXd dFdx = MatrixXd::Zero(input.nVar, input.ncell);
    for (int cell = 0; cell < input.ncell; ++cell) {
        const double dx = input.xface(cell + 1) - input.xface(cell);
        dFdx.col(cell) = (Fc.col(cell + 1) - Fc.col(cell)) / dx;
    }

    return dFdx;
}

MatrixXd& U_correctY(MatrixXd & U)
{
   // 用于修正最后一个组分项，因为实际参与推进的是3个守恒变量+nSpecies-1个组分，最后一个组分项基于组分和为1的约束修正
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

void update_MassFlow_by_time(double t,  InputData& input)
{
    double scale_factor = 1 - exp(-t/input.massflow_buildup_time);
    MatrixXd mdot_profile_by_time = input.mdot_profile.array() * scale_factor;
    MatrixXd dmdot_dx_by_time = input.dmdot_dx.array() * scale_factor;
    MatrixXd dmdot_species_dx_by_time = input.dmdot_species_dx.array() * scale_factor;
    input.mdot_profile_by_time = mdot_profile_by_time;
    input.dmdot_dx_by_time = dmdot_dx_by_time;
    input.dmdot_species_dx_by_time = dmdot_species_dx_by_time;
}

void update_mix_by_time(double t,  InputData& input)
{
    double scale_factor = 1 - exp(-t/input.mix_buildup_time);
    MatrixXd mix_by_time = input.mix_array.array() * scale_factor;
    input.mix_by_time = mix_by_time;
}
