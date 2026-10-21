#ifndef BOUNDARY_CONDITION_H
#define BOUNDARY_CONDITION_H

#include <Eigen/Dense>

#include "preComputation.H"

namespace Cantera
{
class ThermoPhase;
}

/** 从 U 与 local_A 提取 rho, u, p, Y, Ma（Cantera 闭合） */
void rho_u_p_Y_M_extract(const Eigen::MatrixXd & U,
                         double local_A,
                         Cantera::ThermoPhase & gas,
                         double & rho,
                         double & u,
                         double & p,
                         double & Ma,
                         Eigen::MatrixXd & Y);

/**
 * 出口边界守恒列向量（nVar×1）。
 * b=边界点，d=出口左侧内点，a=外部环境 (p_a, c_a, rho_a)。
 * 内点 Ma≥1：外推最后一个物理单元的原始量，并按 A_b/A_d 缩放守恒量；
 * 亚声速：指定 p_a，黎曼线性化 + Cantera 闭合。
 */
Eigen::MatrixXd generateOutletBoundaryVector(const Eigen::MatrixXd & U,
                                             double A_d,
                                             double A_b,
                                             double p_a,
                                             double c_a,
                                             double rho_a,
                                             Cantera::ThermoPhase & gas);

/**
 * 入口边界守恒列向量（nVar×1）。
 * a=外储，b=边界点，d=入口右侧内点。
 * 亚声速：外储–内点 Riemann；超声速：外储全指定。
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

/**
 * 在 N 个物理单元外构造左右各一个边界 ghost 单元。
 * ghost 的面积关于物理边界面对称，从而保持 rhoA/pA 的整体设计。
 */
Eigen::MatrixXd addBoundaryGhostCells(const Eigen::MatrixXd& U,
                                      const InputData& input,
                                      Cantera::ThermoPhase& gas);

/** 对任意列阵左右各做一次线性外推，仅用于补足 NND 重构模板。 */
Eigen::MatrixXd extrapolation(const Eigen::MatrixXd & U);

#endif
