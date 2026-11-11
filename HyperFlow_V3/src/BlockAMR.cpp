#include "BlockAMR.hpp"
#include <AMReX_BoxIterator.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace amrex;
namespace {
constexpr int nv=ns+3;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
State readq(Array4<const Real> a,int i,int j){State q;for(int n=0;n<nv;++n)q[n]=a(i,j,0,n);return q;}
void putq(Array4<Real> a,int i,int j,const State& q){for(int n=0;n<nv;++n)a(i,j,0,n)=q[n];}
State primitive(Gas& gas,double T,double p,const std::string& composition,bool mole,double u=0.,double v=0.) {
    if(mole)gas.setState_TPX(T,p,composition);else gas.setState_TPY(T,p,composition);
    State q;gas.getMassFractions(q.data());const double rho=gas.density();q.head<ns>()*=rho;
    q[mx]=rho*u;q[my]=rho*v;q[energy]=rho*(gas.intEnergy_mass()+0.5*(u*u+v*v));return q;
}
}

BlockAMR::BlockAMR(Json config,std::filesystem::path directory)
    :c(std::move(config)),out(std::move(directory)),
    solution(Cantera::newSolution(c.at("mechanism").get<std::string>(),"","none")),gas(*solution->thermo()),
    chemistry(c.at("mechanism"),c.value("chemistry_threads",8)),mapper(c.at("mechanism").get<std::string>()) {
    cfl=c.value("cfl",0.6);
    if(!(cfl>0 && cfl<=0.8) || !(c.at("end_time").get<double>()>0))throw std::invalid_argument("invalid CFL or final time");
    if(c.value("regrid_interval",8)<1)throw std::invalid_argument("invalid regrid interval");
    const std::array<std::string,ns> names={"H2","O2","H2O","H","O","OH","HO2","H2O2","N2"};
    if(gas.nSpecies()!=ns)throw std::runtime_error("mechanism must have nine species");
    for(int n=0;n<ns;++n)if(gas.speciesName(n)!=names[n])throw std::runtime_error("mechanism species order mismatch");
    state.resize(max_level+1);bcs.resize(nv);
    const auto lo=c.value("boundary_lo",std::vector<std::string>{"reflecting","reflecting"});
    const auto hi=c.value("boundary_hi",std::vector<std::string>{"transmissive","reflecting"});
    if(lo.size()!=2 || hi.size()!=2)throw std::invalid_argument("two boundary conditions required per side");
    mixedBottom=lo[1]=="wedge";
    wedgeStart=c.value("wedge_start",0.);
    if(mixedBottom){
        const double index=(wedgeStart-geom[0].ProbLo(0))/geom[0].CellSize(0);
        if(geom[0].isPeriodic(1) || std::abs(index-std::round(index))>1e-10)
            throw std::invalid_argument("wedge tip must align with a base-grid x face; bottom cannot be periodic");
    }
    for(int d=0;d<2;++d){
        if((lo[d]!="reflecting" && lo[d]!="transmissive" && lo[d]!="inflow" && !(d==1 && lo[d]=="wedge")) ||
           (hi[d]!="reflecting" && hi[d]!="transmissive" && hi[d]!="inflow"))
            throw std::invalid_argument("unsupported boundary type");
        loWall[d]=!geom[0].isPeriodic(d) && (lo[d]=="reflecting" || lo[d]=="wedge");
        hiWall[d]=!geom[0].isPeriodic(d) && hi[d]=="reflecting";
        for(int n=0;n<nv;++n){
            const int reflection=n==(d==0?mx:my)?BCType::reflect_odd:BCType::reflect_even;
            bcs[n].setLo(d,geom[0].isPeriodic(d)?BCType::int_dir:((lo[d]=="inflow" || lo[d]=="wedge")?BCType::ext_dir:(loWall[d]?reflection:BCType::foextrap)));
            bcs[n].setHi(d,geom[0].isPeriodic(d)?BCType::int_dir:(hi[d]=="inflow"?BCType::ext_dir:(hiWall[d]?reflection:BCType::foextrap)));
        }
    }
    const std::string problem=c.at("problem");
    if(problem=="teng_osw"){
        const double speed=c.at("inflow_speed"),angle=c.at("wedge_angle_deg");
        if(!(speed>0 && angle>0 && angle<90))throw std::invalid_argument("invalid wedge inflow speed or angle");
        const double radians=angle*std::acos(-1.)/180.;
        c["u0"]=speed*std::cos(radians);c["v0"]=-speed*std::sin(radians);
    }
    fresh=primitive(gas,c.value("T0",300.),c.value("P0",101325.),"H2:2,O2:1,N2:3.76",true,
                    c.value("u0",0.),c.value("v0",0.));
    hot=primitive(gas,c.value("hot_T",1500.),c.value("hot_P",10.*101325.),"H2:2,O2:1,N2:3.76",true);
    if(problem=="example6" || problem=="example7"){
        fresh=primitive(gas,298.,101325.,"H2:0.02852103,O2:0.226362,N2:0.745117",false);
        hot=primitive(gas,2941.677242,1481999.362037,
            "H:0.000247,O:0.001617,H2O:0.225404,OH:0.014915,O2:0.013336,H2:0.002429,H2O2:0.000002601600,HO2:0.00001857550,N2:0.742031",
            false,problem=="example6"?800.:2000.);
    }else if(problem=="sun_cj")hot=primitive(gas,3000.,90.*101325.,"H2:2,O2:1,N2:3.76",true);
    else if(problem!="uniform" && problem!="entropy_wave" && problem!="blast2d" && problem!="sod_x" && problem!="sod_y" && problem!="teng_osw")
        throw std::invalid_argument("unknown problem");
    std::filesystem::create_directory(out);
    std::ofstream(out/"input.json")<<c.dump(2)<<'\n';
    history.open(out/"history.csv");
    history<<"step,t,dt,finest_level,leaf_cells,stored_cells,dx_min,dy_min,mass,momentum_x,momentum_y,total_energy,regrids,retries\n";
}

State BlockAMR::initial(double x,double y,double dx,double dy){
    const std::string problem=c.at("problem");
    if(problem=="uniform" || problem=="teng_osw")return fresh;
    if(problem=="sun_cj"){
        const double fraction=std::clamp((c.value("hotspot_width",0.002)-(x-0.5*dx))/dx,0.,1.);
        return fraction*hot+(1-fraction)*fresh;
    }
    if(problem=="example6"){
        const double f=std::clamp((0.4-(x-0.5*dx))/dx,0.,1.);return f*hot+(1-f)*fresh;
    }
    if(problem=="sod_x" || problem=="sod_y"){
        const bool axis=problem=="sod_x";const double center=axis?x:y,width=axis?dx:dy;
        const double front=c.value("interface",0.5*geom[0].ProbLength(axis?0:1));
        const double f=std::clamp((front-center+0.5*width)/width,0.,1.);return f*hot+(1-f)*fresh;
    }
    const int quadrature=c.value("initial_subsamples",4);
    if(quadrature<1 || quadrature>32)throw std::invalid_argument("invalid initial quadrature");
    State result=State::Zero();
    const auto center=c.value("center",std::vector<double>{0.3*geom[0].ProbLength(0),0.5*geom[0].ProbLength(1)});
    if(center.size()!=2)throw std::invalid_argument("center must have two coordinates");
    for(int b=0;b<quadrature;++b)for(int a=0;a<quadrature;++a){
        const double xx=x+dx*((a+0.5)/quadrature-0.5),yy=y+dy*((b+0.5)/quadrature-0.5);
        if(problem=="entropy_wave"){
            const double sigma=c.value("sigma",0.08*geom[0].ProbLength(0));
            const double r2=(xx-center[0])*(xx-center[0])+(yy-center[1])*(yy-center[1]);
            const double bump=1+c.value("amplitude",0.2)*std::exp(-r2/(sigma*sigma));
            result+=primitive(gas,c.value("T0",300.)/bump,c.value("P0",101325.),"H2:2,O2:1,N2:3.76",true,c.value("u0",0.),c.value("v0",0.));
        }else{
            bool inside=false;
            if(problem=="example7"){
                const double lower=std::min(yy,geom[0].ProbLength(1)-yy);
                const double front=std::max(0.5,0.25+lower);
                inside=xx<front-32*std::numeric_limits<double>::epsilon()*geom[0].ProbLength(0);
            }else inside=std::hypot(xx-center[0],yy-center[1])<c.value("radius",0.1*geom[0].ProbLength(1));
            result+=inside?hot:fresh;
        }
    }
    return result/(quadrature*quadrature);
}

void BlockAMR::MakeNewLevelFromScratch(int lev,Real,const BoxArray& ba,const DistributionMapping& dm){
    state[lev]=std::make_unique<MultiFab>(ba,dm,nv,0);
    const auto dx=geom[lev].CellSizeArray(),lo=geom[lev].ProbLoArray();
    for(MFIter mfi(*state[lev]);mfi.isValid();++mfi){
        auto a=state[lev]->array(mfi);
        for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const auto iv=bit();putq(a,iv[0],iv[1],initial(lo[0]+(iv[0]+0.5)*dx[0],lo[1]+(iv[1]+0.5)*dx[1],dx[0],dx[1]));}
    }
}
FlowBoundary BlockAMR::boundaryData()const{
    FlowBoundary result;result.mixedBottom=mixedBottom;result.wedgeStart=wedgeStart;
    for(int n=0;n<nv;++n)result.inlet[n]=fresh[n];
    return result;
}
void BlockAMR::fill(int lev,MultiFab& dest,const Levels& data,double t){
    GpuBndryFuncFab<FlowBoundary> boundary(boundaryData());
    PhysBCFunct<GpuBndryFuncFab<FlowBoundary>> fp(geom[lev],bcs,boundary);
    if(lev==0)FillPatchSingleLevel(dest,t,{data[0].get()},{t},0,0,nv,geom[0],fp,0);
    else{
        PhysBCFunct<GpuBndryFuncFab<FlowBoundary>> cp(geom[lev-1],bcs,boundary);
        FillPatchTwoLevels(dest,t,{data[lev-1].get()},{t},{data[lev].get()},{t},0,0,nv,
            geom[lev-1],geom[lev],cp,0,fp,0,refRatio(lev-1),&mapper,bcs,0);
    }
}
void BlockAMR::MakeNewLevelFromCoarse(int lev,Real t,const BoxArray& ba,const DistributionMapping& dm){
    state[lev]=std::make_unique<MultiFab>(ba,dm,nv,0);
    GpuBndryFuncFab<FlowBoundary> boundary(boundaryData());
    PhysBCFunct<GpuBndryFuncFab<FlowBoundary>> cp(geom[lev-1],bcs,boundary),fp(geom[lev],bcs,boundary);
    InterpFromCoarseLevel(*state[lev],t,*state[lev-1],0,0,nv,geom[lev-1],geom[lev],cp,0,fp,0,refRatio(lev-1),&mapper,bcs,0);
}
void BlockAMR::RemakeLevel(int lev,Real t,const BoxArray& ba,const DistributionMapping& dm){
    auto next=std::make_unique<MultiFab>(ba,dm,nv,0);fill(lev,*next,state,t);state[lev]=std::move(next);
}
void BlockAMR::ClearLevel(int lev){state[lev].reset();}
void BlockAMR::average(Levels& data){
    for(int lev=finest_level;lev>0;--lev)average_down(*data[lev],*data[lev-1],geom[lev],geom[lev-1],0,nv,refRatio(lev-1));
}
BlockAMR::Levels BlockAMR::clone(const Levels& data)const{
    Levels copy(max_level+1);
    for(int lev=0;lev<=finest_level;++lev){copy[lev]=std::make_unique<MultiFab>(data[lev]->boxArray(),data[lev]->DistributionMap(),nv,0);MultiFab::Copy(*copy[lev],*data[lev],0,0,nv,0);}
    return copy;
}
iMultiFab BlockAMR::mask(int lev)const{
    if(lev<finest_level)return makeFineMask(*state[lev],state[lev+1]->boxArray(),refRatio(lev),1,0);
    iMultiFab m(state[lev]->boxArray(),state[lev]->DistributionMap(),1,0);m.setVal(1);return m;
}

void BlockAMR::ErrorEst(int lev,TagBoxArray& tags,Real t,int){
    MultiFab ghost(state[lev]->boxArray(),state[lev]->DistributionMap(),nv,2);fill(lev,ghost,state,t);
    const auto dx=geom[lev].CellSizeArray(),lo=geom[lev].ProbLoArray();
    const auto box=c.value("tag_box",std::vector<double>{});
    const auto velocity=c.value("tag_velocity",std::vector<double>{0.,0.});
    const bool onlyBox=c.value("tag_mode",std::string("physics"))=="box";
    if((!box.empty() && box.size()!=4) || velocity.size()!=2 || (onlyBox && box.empty()))throw std::invalid_argument("invalid tagging geometry");
    for(MFIter mfi(ghost);mfi.isValid();++mfi){
        auto a=ghost.const_array(mfi);auto tag=tags.array(mfi);
        for(BoxIterator bit(mfi.validbox());bit.ok();++bit){
            const int i=bit()[0],j=bit()[1];
            const double x=lo[0]+(i+0.5)*dx[0]-velocity[0]*t,y=lo[1]+(j+0.5)*dx[1]-velocity[1]*t;
            bool marked=!box.empty() && x>=box[0] && x<=box[1] && y>=box[2] && y<=box[3];
            if(!onlyBox){
                const State q=readq(a,i,j);const auto p=properties(q,gas);
                for(int d=0;d<2;++d){
                    const int di=d==0,dj=d==1;
                    const auto pl=properties(readq(a,i-di,j-dj),gas),pr=properties(readq(a,i+di,j+dj),gas);
                    const double jump=std::max(std::abs(pl.p-p.p)/std::min(pl.p,p.p),std::abs(pr.p-p.p)/std::min(pr.p,p.p));
                    const double temp=std::max(std::abs(pl.T-p.T)/std::min(pl.T,p.T),std::abs(pr.T-p.T)/std::min(pr.T,p.T));
                    marked=marked || jump>c.value("pressure_tag",0.08) || temp>c.value("temperature_tag",0.08);
                }
                if(c.value("chemistry",true)){
                    properties(q,gas);std::array<double,ns> rates{};solution->kinetics()->getNetProductionRates(rates.data());
                    marked=marked || -rates[0]*gas.molecularWeight(0)/density(q)>c.value("fuel_rate_tag",50.);
                }
            }
            if(marked)tag(i,j,0)=TagBox::SET;
        }
    }
}

void BlockAMR::euler(Levels& data,double h,bool firstOrder){
    Levels ghost(max_level+1),next=clone(data);
    Vector<Fluxes> flux(finest_level+1);
    for(int lev=0;lev<=finest_level;++lev){
        ghost[lev]=std::make_unique<MultiFab>(data[lev]->boxArray(),data[lev]->DistributionMap(),nv,2);
        fill(lev,*ghost[lev],data,time);
        for(int d=0;d<2;++d)flux[lev][d]=std::make_unique<MultiFab>(convert(data[lev]->boxArray(),IntVect::TheDimensionVector(d)),data[lev]->DistributionMap(),nv,0);
    }
    for(int lev=0;lev<=finest_level;++lev){
        const auto dx=geom[lev].CellSizeArray();const Box domain=geom[lev].Domain();
        for(MFIter mfi(*ghost[lev]);mfi.isValid();++mfi){
            auto a=ghost[lev]->const_array(mfi);
            for(int d=0;d<2;++d){
                auto fa=flux[lev][d]->array(mfi);const int di=d==0,dj=d==1;
                for(BoxIterator bit(surroundingNodes(mfi.validbox(),d));bit.ok();++bit){
                    const int i=bit()[0],j=bit()[1];
                    const State qa=readq(a,i-2*di,j-2*dj),qb=readq(a,i-di,j-dj),qc=readq(a,i,j),qd=readq(a,i+di,j+dj);
                    State left=qb+0.5*minmod(qb-qa,qc-qb),right=qc-0.5*minmod(qc-qb,qd-qc);
                    if(firstOrder){left=qb;right=qc;}
                    if(!admissible(left,gas))left=qb;
                    if(!admissible(right,gas))right=qc;
                    State f=hllc(left,right,d==0?mx:my,properties(left,gas),properties(right,gas));
                    const int coordinate=d==0?i:j;
                    const double x=geom[lev].ProbLo(0)+(i+0.5)*dx[0];
                    const bool lowWall=loWall[d] && !(d==1 && mixedBottom && x<wedgeStart);
                    if((coordinate==domain.smallEnd(d) && lowWall) || (coordinate==domain.bigEnd(d)+1 && hiWall[d])){
                        f.head<ns>().setZero();f[d==0?my:mx]=0.;f[energy]=0.;
                    }
                    putq(fa,i,j,f);
                }
            }
        }
        for(int d=0;d<2;++d)flux[lev][d]->OverrideSync(geom[lev].periodicity());
        for(MFIter mfi(*data[lev]);mfi.isValid();++mfi){
            auto a=data[lev]->const_array(mfi),fx=flux[lev][0]->const_array(mfi),fy=flux[lev][1]->const_array(mfi);
            auto b=next[lev]->array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){
                const int i=bit()[0],j=bit()[1];
                State q=readq(a,i,j)-h/dx[0]*(readq(fx,i+1,j)-readq(fx,i,j))-h/dx[1]*(readq(fy,i,j+1)-readq(fy,i,j));
                putq(b,i,j,q);
            }
        }
    }
    // Each Euler stage is a conservative composite-grid operator. Applying
    // reflux at BOTH Heun stages gives the correct RK weights automatically.
    for(int lev=finest_level;lev>0;--lev){
        FluxRegister reg(data[lev]->boxArray(),data[lev]->DistributionMap(),refRatio(lev-1),lev,nv);
        const auto dc=geom[lev-1].CellSizeArray(),df=geom[lev].CellSizeArray();
        for(int d=0;d<2;++d){
            reg.CrseInit(*flux[lev-1][d],d,0,0,nv,-h*dc[1-d]);
            reg.FineAdd(*flux[lev][d],d,0,0,nv,h*df[1-d]);
        }
        reg.ClearInternalBorders(geom[lev-1]);reg.Reflux(*next[lev-1],1.,0,0,nv,geom[lev-1]);++stageCorrections;
    }
    average(next);
    try{repairRoundoff(next);validate(next);}
    catch(const std::domain_error&){
        if(firstOrder)throw;
        // Redo the complete composite Euler stage with first-order face states.
        // Shared conservative fluxes and reflux remain intact; no state clipping.
        ++positivityFallbacks;euler(data,h,true);return;
    }
    data=std::move(next);
}
void BlockAMR::flow(double h){
    const auto start=Clock::now();auto old=clone(state);euler(state,h);euler(state,h);
    for(int lev=0;lev<=finest_level;++lev)MultiFab::LinComb(*state[lev],0.5,*old[lev],0,0.5,*state[lev],0,0,nv,0);
    average(state);validate(state);flowSeconds+=seconds(start);
}
void BlockAMR::react(double h){
    if(!c.value("chemistry",true))return;
    const auto start=Clock::now();
    struct Location{Array4<Real> a;int i,j;};std::vector<Location> locations;Field batch;
    for(int lev=0;lev<=finest_level;++lev){
        auto active=mask(lev);
        for(MFIter mfi(*state[lev]);mfi.isValid();++mfi){
            auto a=state[lev]->array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(ma(i,j,0)){locations.push_back({a,i,j});batch.push_back(readq(a,i,j));}}
        }
    }
    chemistry.advance(batch,h,batch.size());chemCalls+=batch.size();
    for(size_t k=0;k<batch.size();++k)putq(locations[k].a,locations[k].i,locations[k].j,batch[k]);
    average(state);validate(state);chemSeconds+=seconds(start);
}
void BlockAMR::validate(const Levels& data){
    for(int lev=0;lev<=finest_level;++lev)for(MFIter mfi(*data[lev]);mfi.isValid();++mfi){
        auto a=data[lev]->const_array(mfi);
        for(BoxIterator bit(mfi.validbox());bit.ok();++bit){
            const auto q=readq(a,bit()[0],bit()[1]);
            try{properties(q,gas);}
            catch(const std::exception& e){
                std::ostringstream message;message<<std::scientific<<std::setprecision(17)<<e.what()<<" lev="<<lev<<" i="<<bit()[0]<<" j="<<bit()[1]<<" min_Y="<<q.head<ns>().minCoeff()/density(q);
                throw std::domain_error(message.str());
            }
        }
    }
}
void BlockAMR::repairRoundoff(Levels& data){
    // A reflux correction can cancel a provisional coarse trace-species flux
    // to a tiny negative residual (observed Y ~ -1e-52). Redistribute only
    // roundoff-size residuals, preserving EACH species integral, not just rho.
    struct Donor{Array4<Real> a;int i=0,j=0;double volume=0.,mass=0.;};
    std::array<Donor,ns> donors{};
    std::array<double,ns> peak{},deficit{};
    for(int lev=0;lev<=finest_level;++lev){auto active=mask(lev);const auto dx=geom[lev].CellSizeArray();const double volume=dx[0]*dx[1];
        for(MFIter mfi(*data[lev]);mfi.isValid();++mfi){auto a=data[lev]->array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(!ma(i,j,0))continue;
                for(int n=0;n<ns;++n){const double value=a(i,j,0,n);peak[n]=std::max(peak[n],value);
                    if(value*volume>donors[n].mass)donors[n]={a,i,j,volume,value*volume};}
            }
        }
    }
    bool repaired=false;
    for(int lev=0;lev<=finest_level;++lev){auto active=mask(lev);const auto dx=geom[lev].CellSizeArray();const double volume=dx[0]*dx[1];
        for(MFIter mfi(*data[lev]);mfi.isValid();++mfi){auto a=data[lev]->array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(!ma(i,j,0))continue;
                const double rho=density(readq(a,i,j));
                for(int n=0;n<ns;++n){const double value=a(i,j,0,n);if(value>=0)continue;
                    if(!std::isfinite(value) || -value>64*std::numeric_limits<double>::epsilon()*peak[n])
                        throw std::domain_error("materially negative species after composite flow stage");
                    deficit[n]+=-value*volume;maxRoundoffY=std::max(maxRoundoffY,-value/rho);
                    a(i,j,0,n)=0.;++roundoffRepairs;repaired=true;
                }
            }
        }
    }
    for(int n=0;n<ns;++n)if(deficit[n]>0){
        auto& d=donors[n];
        if(!(d.mass>deficit[n]))throw std::domain_error("insufficient positive donor for roundoff redistribution");
        d.a(d.i,d.j,0,n)-=deficit[n]/d.volume;redistributedSpeciesMass[n]+=deficit[n];
    }
    if(repaired)average(data);
}
double BlockAMR::stableDt(){
    double rate=0.;
    for(int lev=0;lev<=finest_level;++lev){const auto dx=geom[lev].CellSizeArray();auto active=mask(lev);
        for(MFIter mfi(*state[lev]);mfi.isValid();++mfi){auto a=state[lev]->const_array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(!ma(i,j,0))continue;
                const auto q=readq(a,i,j);const auto p=properties(q,gas);const double rho=density(q);
                rate=std::max(rate,(std::abs(q[mx]/rho)+p.a)/dx[0]+(std::abs(q[my]/rho)+p.a)/dx[1]);
            }
        }
    }
    return cfl/rate;
}
std::array<double,nv> BlockAMR::totals(bool absolute)const{
    std::array<long double,nv> sum{};
    for(int lev=0;lev<=finest_level;++lev){auto active=mask(lev);const auto dx=geom[lev].CellSizeArray();
        for(MFIter mfi(*state[lev]);mfi.isValid();++mfi){auto a=state[lev]->const_array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(mfi.validbox());bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(ma(i,j,0))for(int n=0;n<nv;++n){const double value=a(i,j,0,n);sum[n]+=static_cast<long double>(absolute?std::abs(value):value)*dx[0]*dx[1];}}
        }
    }
    std::array<double,nv> result{};for(int n=0;n<nv;++n)result[n]=sum[n];return result;
}
long long BlockAMR::leafCount()const{
    long long count=0;
    for(int lev=0;lev<=finest_level;++lev){long long cells=state[lev]->boxArray().numPts();if(lev<finest_level)cells-=state[lev+1]->boxArray().numPts()/4;count+=cells;}
    return count;
}
void BlockAMR::log(double dt){
    long long stored=0;for(int lev=0;lev<=finest_level;++lev)stored+=state[lev]->boxArray().numPts();
    const auto sums=totals();double mass=0.;for(int n=0;n<ns;++n)mass+=sums[n];
    const auto dx=geom[finest_level].CellSizeArray();maxLeaves=std::max(maxLeaves,leafCount());
    history<<std::setprecision(17)<<steps<<','<<time<<','<<dt<<','<<finest_level<<','<<leafCount()<<','<<stored<<','<<dx[0]<<','<<dx[1]<<','<<mass<<','<<sums[mx]<<','<<sums[my]<<','<<sums[energy]<<','<<regrids<<','<<retries<<'\n';history.flush();
}
void BlockAMR::regridChecked(){
    const auto start=Clock::now();average(state);const auto before=totals(),scale=totals(true);
    regrid(0,time);average(state);validate(state);const auto after=totals();
    for(int n=0;n<nv;++n)maxRemapDefect=std::max(maxRemapDefect,std::abs(after[n]-before[n])/std::max(scale[n],1e-300));
    if(maxRemapDefect>1e-8)throw std::runtime_error("nonconservative remap");
    ++regrids;regridSeconds+=seconds(start);
}
void BlockAMR::snapshot(int frame){
    std::ofstream csv(out/("state_"+std::to_string(frame)+".csv"));
    csv<<"t,level,i,j,x,y,dx,dy,rho,u,v,p,p_atm,T,rhoE";
    for(int n=0;n<ns;++n)csv<<",Y_"<<gas.speciesName(n);
    csv<<'\n';csv<<std::setprecision(17);
    std::ofstream boxes(out/("boxes_"+std::to_string(frame)+".csv"));boxes<<"level,x_left,x_right,y_bottom,y_top,dx,dy\n";boxes<<std::setprecision(17);
    for(int lev=0;lev<=finest_level;++lev){auto active=mask(lev);const auto dx=geom[lev].CellSizeArray(),lo=geom[lev].ProbLoArray();
        for(MFIter mfi(*state[lev]);mfi.isValid();++mfi){
            const auto box=mfi.validbox();boxes<<lev<<','<<lo[0]+box.smallEnd(0)*dx[0]<<','<<lo[0]+(box.bigEnd(0)+1)*dx[0]<<','<<lo[1]+box.smallEnd(1)*dx[1]<<','<<lo[1]+(box.bigEnd(1)+1)*dx[1]<<','<<dx[0]<<','<<dx[1]<<'\n';
            auto a=state[lev]->const_array(mfi);auto ma=active.const_array(mfi);
            for(BoxIterator bit(box);bit.ok();++bit){const int i=bit()[0],j=bit()[1];if(!ma(i,j,0))continue;
                const auto q=readq(a,i,j);const auto p=properties(q,gas);const double rho=density(q);
                csv<<time<<','<<lev<<','<<i<<','<<j<<','<<lo[0]+(i+0.5)*dx[0]<<','<<lo[1]+(j+0.5)*dx[1]<<','<<dx[0]<<','<<dx[1]<<','<<rho<<','<<q[mx]/rho<<','<<q[my]/rho<<','<<p.p<<','<<p.p/101325.<<','<<p.T<<','<<q[energy];
                for(int n=0;n<ns;++n)csv<<','<<q[n]/rho;
                csv<<'\n';
            }
        }
    }
    if(c.value("write_plotfiles",true)){
        Vector<std::string> names;for(int n=0;n<ns;++n)names.push_back("rhoY_"+gas.speciesName(n));
        names.push_back("rhou");names.push_back("rhov");names.push_back("rhoE");
        Vector<const MultiFab*> data;Vector<int> step(finest_level+1,steps);
        for(int lev=0;lev<=finest_level;++lev)data.push_back(state[lev].get());
        WriteMultiLevelPlotfile((out/("plt_"+std::to_string(frame))).string(),finest_level+1,data,names,geom,time,step,ref_ratio);
    }
}
void BlockAMR::run(){
    const auto start=Clock::now();InitFromScratch(0.);average(state);validate(state);
    const auto initialTotals=totals();snapshot(0);log(0.);
    amrex::Print()<<"2D block AMR: CFL="<<cfl<<" levels="<<finest_level+1<<" leaves="<<leafCount()<<" chemistry_workers="<<chemistry.threads()<<'\n';
    std::vector<double> targets=c.value("output_times",std::vector<double>{c.at("end_time").get<double>()});
    const double end=c.at("end_time");double previous=0.;
    for(double target:targets){if(!(target>previous && target<=end))throw std::invalid_argument("invalid output times");previous=target;}
    if(targets.empty() || std::abs(targets.back()-end)>1e-14*end)throw std::invalid_argument("last output must equal final time");
    double minDt=std::numeric_limits<double>::infinity(),maxDt=0.;int frame=0;
    for(double target:targets){
        while(time<target){
            if(steps>=c.value("max_steps",10000000L))throw std::runtime_error("maximum step count reached");
            double dt=std::min(target-time,stableDt());if(c.value("max_dt",0.)>0)dt=std::min(dt,c.at("max_dt").get<double>());
            const auto backup=clone(state);bool accepted=false;
            for(int attempt=0;attempt<18;++attempt){
                try{flow(0.5*dt);react(dt);flow(0.5*dt);accepted=true;break;}
                catch(const std::exception& error){state=clone(backup);++retries;amrex::Print()<<"rejected step at t="<<time<<": "<<error.what()<<'\n';dt*=0.5;}
            }
            if(!accepted || !(dt>0) || time+dt==time)throw std::runtime_error("time advancement failed");
            time=dt>=target-time?target:time+dt;++steps;minDt=std::min(minDt,dt);maxDt=std::max(maxDt,dt);
            if(max_level>0 && c.value("adaptive",true) && steps%c.value("regrid_interval",8)==0 && time<end)regridChecked();
            log(dt);
            if(steps%50==0)amrex::Print()<<std::setprecision(10)<<"t="<<time<<" steps="<<steps<<" leaves="<<leafCount()<<" finest="<<finest_level<<'\n';
        }
        snapshot(++frame);amrex::Print()<<"output t="<<time<<" steps="<<steps<<" leaves="<<leafCount()<<'\n';
    }
    const auto finalTotals=totals();
    Json result={{"completed",true},{"final_time",time},{"cfl",cfl},{"steps",steps},{"retries",retries},{"finest_level",finest_level},
        {"leaf_cells",leafCount()},{"max_leaf_cells",maxLeaves},{"full_finest_cells",geom[finest_level].Domain().numPts()},
        {"dx_min",geom[finest_level].CellSize(0)},{"dy_min",geom[finest_level].CellSize(1)},
        {"regrids",regrids},{"max_remap_relative_defect",maxRemapDefect},{"reflux_stage_interfaces",stageCorrections},
        {"positivity_first_order_stages",positivityFallbacks},
        {"roundoff_redistributions",roundoffRepairs},{"max_repaired_negative_Y",maxRoundoffY},
        {"redistributed_species_integrals",redistributedSpeciesMass},
        {"chemistry_cell_calls",chemCalls},{"chemistry_threads",chemistry.threads()},{"last_chemistry_team",chemistry.lastTeamSize()},
        {"chemistry_seconds",chemSeconds},{"flow_seconds",flowSeconds},{"regrid_seconds",regridSeconds},{"wall_seconds",seconds(start)},
        {"min_dt",minDt},{"max_dt",maxDt},{"initial_conserved_integrals",initialTotals},{"final_conserved_integrals",finalTotals},
        {"time_subcycling",false},{"refinement_ratio",{2,2}}};
    std::ofstream(out/"summary.json")<<result.dump(2)<<'\n';
}
