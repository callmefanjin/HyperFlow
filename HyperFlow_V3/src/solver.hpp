#pragma once
#include <Eigen/Core>
#include <cantera/base/Solution.h>
#include <cantera/thermo/ThermoPhase.h>
#include <cantera/kinetics/Kinetics.h>
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

constexpr int ns = 9, mx = ns, my = ns+1, energy = ns+2;
using State = Eigen::Matrix<double,ns+3,1>;
using Field = std::vector<State>;
using Gas = Cantera::ThermoPhase;
inline double density(const State& u) { return u.head<ns>().sum(); }
inline double internalEnergy(const State& u) {
    const double rho = density(u);
    return u[energy]/rho - 0.5*(std::pow(u[mx]/rho,2)+std::pow(u[my]/rho,2));
}
struct Props { double p, a, T; };
Props properties(const State& u, Gas& gas);
State minmod(const State&,const State&);
bool admissible(const State&,Gas&);
State hllc(const State&,const State&,int,Props,Props);
class Chemistry {
public:
    explicit Chemistry(std::shared_ptr<Cantera::Solution> solution);
    ~Chemistry();
    Chemistry(const Chemistry&) = delete;
    Chemistry& operator=(const Chemistry&) = delete;
    void advance(State& u, double dt);
private:
    static int rhs(double, N_Vector y, N_Vector dy, void* data);
    std::shared_ptr<Cantera::Solution> sol;
    double rho{}, e{};
    SUNContext ctx{};
    N_Vector y{};
    SUNMatrix jac{};
    SUNLinearSolver linear{};
    void* cv{};
};
