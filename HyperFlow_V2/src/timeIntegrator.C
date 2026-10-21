#include "timeIntegrator.H"
#include "coreMethod.H"
#include "sourceTerm.H"
#include "boundaryCondition.H"
#include "cantera/core.h"
using namespace Eigen;
using namespace Cantera;

FlowWorkBuffers::FlowWorkBuffers(int nVar, int nface)
    : dFdx(nVar, nface)
    , source(nVar, nface)
    , rhs(nVar, nface)
    , k1(nVar, nface)
    , k2(nVar, nface)
    , k3(nVar, nface)
    , U_1(nVar, nface)
    , U_2(nVar, nface)
    , U_out(nVar, nface)
{
}

void flow_rhs(const MatrixXd & U,
              ThermoPhase & gas,
              const InputData & input,
              bool isInjection,
              FlowWorkBuffers & buf,
              const int spatial_order)
{
    auto [rhoA, u, e, Y] = baseVar_From_conserVar(U);
    auto [pA, T, M] = PTM_From_baseVar(rhoA, u, e, Y, gas);
    (void)T; (void)M;
    buf.dFdx = dFdx_calculation(
        U, gas, input, spatial_order);
    sourceTerm_calculation(pA, rhoA, u, input, isInjection, buf.source);
    buf.rhs = buf.source - buf.dFdx;
}

void SSPRK3_step(const MatrixXd & U,
                 ThermoPhase & gas,
                 const InputData & input,
                 double dt,
                 bool isInjection,
                 FlowWorkBuffers & buf,
                 const int spatial_order)
{
    flow_rhs(U, gas, input, isInjection, buf, spatial_order);
    buf.k1 = buf.rhs;
    buf.U_1 = U + buf.k1 * dt;
    U_correctY(buf.U_1);

    flow_rhs(buf.U_1, gas, input, isInjection, buf, spatial_order);
    buf.k2 = buf.rhs;
    buf.U_2 = 0.75 * U + 0.25 * buf.U_1 + 0.25 * dt * buf.k2;
    U_correctY(buf.U_2);

    flow_rhs(buf.U_2, gas, input, isInjection, buf, spatial_order);
    buf.k3 = buf.rhs;
    buf.U_out = (1.0 / 3.0) * U + (2.0 / 3.0) * buf.U_2 + (2.0 / 3.0) * dt * buf.k3;
    U_correctY(buf.U_out);
}
