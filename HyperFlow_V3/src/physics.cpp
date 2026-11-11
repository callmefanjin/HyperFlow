#include "solver.hpp"
#include <algorithm>

Props properties(const State& u, Gas& gas) {
    const double rho = density(u);
    if (!u.allFinite() || !(rho > 0) || u.head<ns>().minCoeff() < 0)
        throw std::domain_error("inadmissible conserved state");
    Eigen::Matrix<double,ns,1> Y = u.head<ns>()/rho;
    gas.setMassFractions_NoNorm(Y.data());
    gas.setState_UV(internalEnergy(u), 1/rho, 1e-11);
    Props s{gas.pressure(),gas.soundSpeed(),gas.temperature()};
    if (!(s.T > 0 && s.p > 0 && s.a > 0 && std::isfinite(s.a)))
        throw std::domain_error("invalid EOS state");
    return s;
}

State hllc(const State& l, const State& r, int n, Props pl, Props pr) {
    const double rl=density(l), rr=density(r), ul=l[n]/rl, ur=r[n]/rr;
    const double sl=std::min(ul-pl.a,ur-pr.a), sr=std::max(ul+pl.a,ur+pr.a);
    auto flux = [n](const State& q, double v, double p) {
        State f=v*q;
        f[n]+=p;
        f[energy]+=p*v;
        return f;
    };
    if (sl>=0) return flux(l,ul,pl.p);
    if (sr<=0) return flux(r,ur,pr.p);
    const double dl=rl*(sl-ul), dr=rr*(sr-ur);
    const double sm=(pr.p-pl.p+dl*ul-dr*ur)/(dl-dr);
    const bool left=sm>=0;
    const auto& q=left?l:r;
    const double v=left?ul:ur, s=left?sl:sr, p=left?pl.p:pr.p;
    State star=q*((s-v)/(s-sm));
    const double rs=density(star);
    star[n]=rs*sm;
    star[energy]+=rs*(sm-v)*(sm+p/(density(q)*(s-v)));
    return flux(q,v,p)+s*(star-q);
}

State minmod(const State& a,const State& b) {
    return a.binaryExpr(b,[](double x,double y){
        return x*y>0 ? std::copysign(std::min(std::abs(x),std::abs(y)),x) : 0.;
    });
}

bool admissible(const State& q,Gas& gas) {
    try {properties(q,gas);return true;}catch(const std::exception&){return false;}
}
