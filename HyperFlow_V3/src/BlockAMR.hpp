#pragma once
#include "ConservativeInterp.hpp"
#include "chemistry_batch.hpp"
#include "FlowBoundary.hpp"
#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

class BlockAMR final : public amrex::AmrCore {
public:
    using Json=nlohmann::json;
    BlockAMR(Json config,std::filesystem::path directory);
    void run();
private:
    using Levels=amrex::Vector<std::unique_ptr<amrex::MultiFab>>;
    using Fluxes=std::array<std::unique_ptr<amrex::MultiFab>,2>;
    Json c;
    std::filesystem::path out;
    std::shared_ptr<Cantera::Solution> solution;
    Gas& gas;
    ChemistryBatch chemistry;
    ConservativeInterp mapper;
    Levels state;
    amrex::Vector<amrex::BCRec> bcs;
    std::array<int,2> loWall{},hiWall{};
    bool mixedBottom=false;
    double wedgeStart=0.;
    double time=0.,cfl=0.6,chemSeconds=0.,flowSeconds=0.,regridSeconds=0.,maxRemapDefect=0.;
    long steps=0,retries=0,regrids=0,chemCalls=0,stageCorrections=0,positivityFallbacks=0;
    long long maxLeaves=0;
    long roundoffRepairs=0;
    double maxRoundoffY=0.;
    std::array<double,ns> redistributedSpeciesMass{};
    std::ofstream history;
    State fresh,hot;
    void MakeNewLevelFromScratch(int,amrex::Real,const amrex::BoxArray&,const amrex::DistributionMapping&) override;
    void MakeNewLevelFromCoarse(int,amrex::Real,const amrex::BoxArray&,const amrex::DistributionMapping&) override;
    void RemakeLevel(int,amrex::Real,const amrex::BoxArray&,const amrex::DistributionMapping&) override;
    void ClearLevel(int) override;
    void ErrorEst(int,amrex::TagBoxArray&,amrex::Real,int) override;
    void fill(int,amrex::MultiFab&,const Levels&,double);
    FlowBoundary boundaryData()const;
    void average(Levels&);
    Levels clone(const Levels&) const;
    amrex::iMultiFab mask(int) const;
    State initial(double,double,double,double);
    void flow(double);
    void euler(Levels&,double,bool firstOrder=false);
    void react(double);
    void validate(const Levels&);
    void repairRoundoff(Levels&);
    double stableDt();
    std::array<double,ns+3> totals(bool absolute=false) const;
    long long leafCount() const;
    void snapshot(int);
    void log(double);
    void regridChecked();
};
