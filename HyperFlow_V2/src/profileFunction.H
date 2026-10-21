#ifndef PROFILE_FUNCTION_H
#define PROFILE_FUNCTION_H

#include <tuple>

#include <Eigen/Dense>

#include "preComputation.H"

/** 质量流剖面 ṁ(x)、dṁ/dx 及物种质量添加率 dṁ_species/dx；喷注点 Hermite 平滑过渡（过渡长度 0.05 m） */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> mdot_profile_calculation(
    double massflow_inlet,
    double massflow_injection,
    double injection_location,
    const Eigen::MatrixXd & Y_injection,
    int nSpecies,
    const Eigen::VectorXd & xcell,
    const Eigen::VectorXd & xface);

/** 掺混因子 η_mix(x)：注入前 0，注入后 L_m 为 1，中间 (1-e^{-αξ})/(1-e^{-α})，ξ=(x-x_inj)/L_m */
void mix_profile_calculation(InputData & input);

/** 截面积剖面 A(x)、dA/dx、dlnA/dx：等直管道 → 线性扩张 → 出口段（三次 Hermite 型曲线） */
void A_profile_calculation(InputData & input);

#endif
