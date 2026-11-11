#pragma once
#include "solver.hpp"
#include <AMReX_Interpolater.H>

// One common limiter for all conservative components and all four children.
// This preserves the parent average even when EOS/positivity limits a slope.
class ConservativeInterp final : public amrex::Interpolater {
public:
    explicit ConservativeInterp(const std::string& mechanism);
    amrex::Box CoarseBox(const amrex::Box& b,int r) override;
    amrex::Box CoarseBox(const amrex::Box& b,const amrex::IntVect& r) override;
    void interp(const amrex::FArrayBox&,int,amrex::FArrayBox&,int,int,
        const amrex::Box&,const amrex::IntVect&,const amrex::Geometry&,
        const amrex::Geometry&,const amrex::Vector<amrex::BCRec>&,int,int,amrex::RunOn) override;
private:
    std::vector<std::shared_ptr<Cantera::Solution>> phases_;
};
