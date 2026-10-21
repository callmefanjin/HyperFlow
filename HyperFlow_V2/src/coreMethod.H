#ifndef CORE_METHOD_H
#define CORE_METHOD_H

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <vector>

#include "preComputation.H"

namespace Cantera
{
class ThermoPhase;
class Solution;
}

/** NND 限制器 */
Eigen::MatrixXd minmod(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b);

/** NND 重构：为每个相邻控制体共享界面生成左、右状态 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
difference_format_NND(const Eigen::MatrixXd & U_dummy, int ndummy, const InputData & input);

/** 一阶分片常数重构：为耗散型稳态预迭代/诊断生成左右状态 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
difference_format_first_order(const Eigen::MatrixXd & U_dummy, int ndummy,
                              const InputData & input);

/** 守恒量 U → (rhoA, u, e, Y) */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
baseVar_From_conserVar(const Eigen::MatrixXd & U);

/** 基本量 (rhoA, u, e, Y) → (p, T, M) */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
PTM_From_baseVar(const Eigen::MatrixXd & rhoA,
                 const Eigen::MatrixXd & u,
                 const Eigen::MatrixXd & e,
                 const Eigen::MatrixXd & Y,
                 Cantera::ThermoPhase & gas);

/** HLLC 数值通量；rho_L/R 在调用处可为 rhoA */
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

/** FVM 通量散度：每个内部控制体采用共享界面的 (F_right-F_left)/dx */
Eigen::MatrixXd dFdx_calculation(const Eigen::MatrixXd & U,
                                 Cantera::ThermoPhase & gas,
                                 const InputData & input,
                                 int spatial_order = 2);

/** 用 ΣY_k=1 修正最后一个组分（ρA Y_k），原地更新 U 并返回引用 */
Eigen::MatrixXd & U_correctY(Eigen::MatrixXd & U);

void update_MassFlow_by_time(double t, InputData& input);
void update_mix_by_time(double t, InputData& input);

#endif
