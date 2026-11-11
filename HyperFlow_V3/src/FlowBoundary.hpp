#pragma once
#include "solver.hpp"
#include <AMReX_PhysBCFunct.H>

// Fixed premixed inflow and a flat wall beginning at a resolved x-face.
// The wedge geometry is represented in coordinates rotated along its surface.
struct FlowBoundary {
    amrex::GpuArray<amrex::Real,ns+3> inlet{};
    bool mixedBottom=false;
    double wedgeStart=0.;
    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect& iv,amrex::Array4<amrex::Real> const& a,
        int dcomp,int ncomp,amrex::GeometryData const& geometry,amrex::Real,
        const amrex::BCRec* bcr,int bcomp,int original)const {
        const auto domain=geometry.Domain();
        const bool below=iv[1]<domain.smallEnd(1);
        for(int n=0;n<ncomp;++n){
            const auto& bc=bcr[bcomp+n];bool fixed=false;
            for(int d=0;d<2;++d){
                const bool mixed=mixedBottom && d==1 && below;
                fixed=fixed || (iv[d]<domain.smallEnd(d) && bc.lo(d)==amrex::BCType::ext_dir && !mixed);
                fixed=fixed || (iv[d]>domain.bigEnd(d) && bc.hi(d)==amrex::BCType::ext_dir);
            }
            if(fixed){a(iv[0],iv[1],0,dcomp+n)=inlet[original+n];continue;}
            if(mixedBottom && below){
                const auto dx=geometry.CellSize(),lo=geometry.ProbLo();
                const double x=lo[0]+(iv[0]-domain.smallEnd(0)+0.5)*dx[0];
                const bool wall=x>=wedgeStart;
                const int j=wall?2*domain.smallEnd(1)-iv[1]-1:domain.smallEnd(1);
                const int i=amrex::min(amrex::max(iv[0],domain.smallEnd(0)),domain.bigEnd(0));
                double value=a(i,j,0,dcomp+n);
                if(wall && original+n==my)value=-value;
                a(iv[0],iv[1],0,dcomp+n)=value;
            }
        }
    }
};
