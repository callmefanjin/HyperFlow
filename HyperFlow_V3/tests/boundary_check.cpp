#include "FlowBoundary.hpp"
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <iostream>
#include <omp.h>

int main(int argc,char** argv){
    omp_set_dynamic(0);omp_set_num_threads(4);
    amrex::Initialize(argc,argv,false);
    int result=0;
    try{
        using namespace amrex;
        const Box domain(IntVect(0,0),IntVect(7,3));
        const RealBox rb({-.002,0.},{.006,.004});
        const int periodic[2]={0,0};Geometry geom(domain,&rb,0,periodic);
        BoxArray ba(domain);DistributionMapping dm(ba);MultiFab data(ba,dm,ns+3,2);data.setVal(-1.e100);
        auto a=data.array(0);
        auto value=[](int i,int j,int n){return 1000.*n+10.*j+i+1.;};
        for(BoxIterator bit(domain);bit.ok();++bit)for(int n=0;n<ns+3;++n)a(bit()[0],bit()[1],0,n)=value(bit()[0],bit()[1],n);
        Vector<BCRec> bcs(ns+3);
        for(auto& bc:bcs){bc.setLo(0,BCType::ext_dir);bc.setHi(0,BCType::foextrap);bc.setLo(1,BCType::ext_dir);bc.setHi(1,BCType::ext_dir);}
        FlowBoundary fill;fill.mixedBottom=true;fill.wedgeStart=0.;for(int n=0;n<ns+3;++n)fill.inlet[n]=20000.+n;
        GpuBndryFuncFab<FlowBoundary> f(fill);PhysBCFunct<GpuBndryFuncFab<FlowBoundary>> physical(geom,bcs,f);
        physical(data,0,ns+3,IntVect(2),0.,0);
        auto check=[](double observed,double expected){if(observed!=expected)throw std::runtime_error("boundary value mismatch");};
        for(int n=0;n<ns+3;++n){
            check(a(-1,1,0,n),fill.inlet[n]);check(a(3,4,0,n),fill.inlet[n]);
            check(a(-1,-1,0,n),fill.inlet[n]);check(a(8,2,0,n),value(7,2,n));
            // x<0: outflow copies nearest interior row, including the 2nd ghost.
            check(a(0,-1,0,n),value(0,0,n));check(a(0,-2,0,n),value(0,0,n));
            // x>=0: slip wall mirrors rows and changes ONLY normal momentum.
            const double sign=n==my?-1.:1.;
            check(a(3,-1,0,n),sign*value(3,0,n));check(a(3,-2,0,n),sign*value(3,1,n));
        }
        std::cout<<"PASS fixed left/top inflow, right outflow, pre-wedge bottom outflow, post-tip slip wall, corners\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
    amrex::Finalize();return result;
}
