#include "solver.hpp"
#include <algorithm>
#include <numeric>
#include <string>

static void check(int flag,const char* operation) {
    if (flag<0) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(flag));
}

Chemistry::Chemistry(std::shared_ptr<Cantera::Solution> solution):sol(std::move(solution)) {
    check(SUNContext_Create(SUN_COMM_NULL,&ctx),"SUNContext_Create");
    y=N_VNew_Serial(ns-1,ctx);
    if (!y) throw std::bad_alloc();
    N_VConst(0,y);
    cv=CVodeCreate(CV_BDF,ctx);
    if (!cv) throw std::bad_alloc();
    check(CVodeInit(cv,rhs,0,y),"CVodeInit");
    check(CVodeSetUserData(cv,this),"CVodeSetUserData");
    check(CVodeSStolerances(cv,1e-8,1e-15),"CVodeSStolerances");
    jac=SUNDenseMatrix(ns-1,ns-1,ctx);
    if (!jac) throw std::bad_alloc();
    linear=SUNLinSol_Dense(y,jac,ctx);
    if (!linear) throw std::bad_alloc();
    check(CVodeSetLinearSolver(cv,linear,jac),"CVodeSetLinearSolver");
    check(CVodeSetMaxNumSteps(cv,5000),"CVodeSetMaxNumSteps");
}

Chemistry::~Chemistry() {
    CVodeFree(&cv);
    SUNLinSolFree(linear);
    SUNMatDestroy(jac);
    N_VDestroy(y);
    SUNContext_Free(&ctx);
}

int Chemistry::rhs(double,N_Vector value,N_Vector derivative,void* data) {
    auto& c=*static_cast<Chemistry*>(data);
    try {
        std::array<double,ns> Y{}, rates{};
        std::copy_n(N_VGetArrayPointer(value),ns-1,Y.begin());
        Y.back()=1-std::accumulate(Y.begin(),Y.end()-1,0.);
        auto& gas=*c.sol->thermo();
        gas.setMassFractions_NoNorm(Y.data());
        gas.setState_UV(c.e,1/c.rho,1e-11);
        c.sol->kinetics()->getNetProductionRates(rates.data());
        for (int k=0;k<ns-1;++k) {
            const double r=rates[k]*gas.molecularWeight(k)/c.rho;
            if (!std::isfinite(r)) return 1;
            NV_Ith_S(derivative,k)=r;
        }
        return 0;
    } catch (const std::exception&) { return 1; }
}

void Chemistry::advance(State& u,double dt) {
    if (!(dt>0) || !std::isfinite(dt)) throw std::invalid_argument("invalid chemistry interval");
    // Reset the UV inversion starting guess for every cell. Otherwise the
    // previous cell handled by this worker can affect roundoff/iteration paths.
    sol->thermo()->setTemperature(1000.);
    properties(u,*sol->thermo());
    rho=density(u);
    e=internalEnergy(u);
    for (int k=0;k<ns-1;++k) NV_Ith_S(y,k)=u[k]/rho;
    check(CVodeReInit(cv,0,y),"CVodeReInit");
    check(CVodeSetStopTime(cv,dt),"CVodeSetStopTime");
    double reached=0;
    check(CVode(cv,dt,y,&reached,CV_NORMAL),"CVode");
    if (std::abs(reached-dt)>1e-12*dt) throw std::runtime_error("incomplete chemistry interval");
    for (int k=0;k<ns-1;++k) {
        const double v=NV_Ith_S(y,k);
        if (!std::isfinite(v) || v< -1e-13) throw std::domain_error("invalid accepted species");
        u[k]=rho*std::max(0.,v);
    }
    u[ns-1]=rho-u.head<ns-1>().sum();  // Dependent nitrogen closes total mass.
    properties(u,*sol->thermo());
}
