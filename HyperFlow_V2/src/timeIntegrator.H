#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <Eigen/Dense>

#include "preComputation.H"

namespace Cantera
{
class ThermoPhase;
}

/** 时间推进预分配工作区，避免热循环内反复分配整场矩阵 */
struct FlowWorkBuffers
{
    Eigen::MatrixXd dFdx;
    Eigen::MatrixXd source;
    Eigen::MatrixXd rhs;
    Eigen::MatrixXd k1;
    Eigen::MatrixXd k2;
    Eigen::MatrixXd k3;
    Eigen::MatrixXd U_1;
    Eigen::MatrixXd U_2;
    Eigen::MatrixXd U_out;

    FlowWorkBuffers(int nVar, int nface);
};

/** 流动右端项：buf.rhs = S - div(F)，其中 div(F) 为 FVM 共享面通量净流出 */
void flow_rhs(const Eigen::MatrixXd & U,
              Cantera::ThermoPhase & gas,
              const InputData & input,
              bool isInjection,
              FlowWorkBuffers & buf,
              int spatial_order = 2);

/**
 * 三阶 SSP-RK3 推进一步。
 * 结束后新状态在 buf.U_out；末级斜率在 buf.k3。
 */
void SSPRK3_step(const Eigen::MatrixXd & U,
                 Cantera::ThermoPhase & gas,
                 const InputData & input,
                 double dt,
                 bool isInjection,
                 FlowWorkBuffers & buf,
                 int spatial_order = 2);

#endif
