#include "ConservativeInterp.hpp"
#include <AMReX_BoxIterator.H>
#include <AMReX_FArrayBox.H>
#include <omp.h>

ConservativeInterp::ConservativeInterp(const std::string& mechanism){
    // AMReX invokes Interpolater::interp inside OpenMP regions. Never share
    // the flow phase, a kinetics cache, or an EOS inversion state across threads.
    for(int n=0;n<omp_get_max_threads();++n)phases_.push_back(Cantera::newSolution(mechanism,"","none"));
}

amrex::Box ConservativeInterp::CoarseBox(const amrex::Box& b,int r) {
    return amrex::grow(amrex::coarsen(b,r),1);
}
amrex::Box ConservativeInterp::CoarseBox(const amrex::Box& b,const amrex::IntVect& r) {
    return amrex::grow(amrex::coarsen(b,r),1);
}
void ConservativeInterp::interp(const amrex::FArrayBox& coarse,int cc,amrex::FArrayBox& fine,
    int fc,int nc,const amrex::Box& region,const amrex::IntVect& ratio,
    const amrex::Geometry&,const amrex::Geometry&,const amrex::Vector<amrex::BCRec>&,
    int,int,amrex::RunOn) {
    if(nc!=ns+3 || ratio[0]!=2 || ratio[1]!=2)
        throw std::invalid_argument("conservative interpolation requires all components and ratio (2,2)");
    Gas& gas_=*phases_.at(omp_get_thread_num())->thermo();
    auto c=coarse.const_array(); auto f=fine.array();
    auto read=[&](int i,int j) {State q;for(int n=0;n<nc;++n)q[n]=c(i,j,0,cc+n);return q;};
    for(amrex::BoxIterator bit(amrex::coarsen(region,ratio));bit.ok();++bit) {
        const int i=bit()[0],j=bit()[1];
        const State q=read(i,j);
        const State sx=minmod(q-read(i-1,j),read(i+1,j)-q);
        const State sy=minmod(q-read(i,j-1),read(i,j+1)-q);
        double theta=1.;
        bool accepted=false;
        for(int attempt=0;attempt<48;++attempt) {
            bool good=true;
            for(int b=0;b<2 && good;++b)for(int a=0;a<2 && good;++a) {
                State child=q+theta*((a?0.25:-0.25)*sx+(b?0.25:-0.25)*sy);
                good=admissible(child,gas_);
            }
            if(good){accepted=true;break;}
            theta*=0.5;
        }
        if(!accepted){theta=0.;properties(q,gas_);}
        for(int b=0;b<2;++b)for(int a=0;a<2;++a) {
            amrex::IntVect iv(AMREX_D_DECL(2*i+a,2*j+b,0));
            if(!region.contains(iv))continue;
            const State child=q+theta*((a?0.25:-0.25)*sx+(b?0.25:-0.25)*sy);
            for(int n=0;n<nc;++n)f(iv[0],iv[1],0,fc+n)=child[n];
        }
    }
}
