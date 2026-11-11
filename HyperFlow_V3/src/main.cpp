#include "BlockAMR.hpp"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <iostream>
#include <omp.h>

int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: hyperflow2d case.json NEW_OUTPUT_DIRECTORY\n";return 2;}
    try {
        std::ifstream in(argv[1]);
        if(!in)throw std::runtime_error("cannot open case input");
        auto c=nlohmann::json::parse(in);
        const std::filesystem::path output=argv[2];
        if(std::filesystem::exists(output))throw std::runtime_error("output directory already exists");
        const auto nc=c.at("base_cells").get<std::vector<int>>();
        const auto length=c.at("length").get<std::vector<double>>();
        const auto origin=c.value("origin",std::vector<double>{0.,0.});
        const auto periodic=c.value("periodic",std::vector<int>{0,0});
        const int levels=c.value("max_level",2),blocking=c.value("blocking_factor",4);
        if(nc.size()!=2 || length.size()!=2 || origin.size()!=2 || periodic.size()!=2 || levels<0 || levels>12 || blocking<2)
            throw std::invalid_argument("invalid 2D mesh configuration");
        for(int d=0;d<2;++d)if(nc[d]<blocking || nc[d]%blocking || !(length[d]>0) || (periodic[d]!=0 && periodic[d]!=1))
            throw std::invalid_argument("base cells must be divisible by blocking_factor and lengths positive");
        omp_set_dynamic(0);omp_set_num_threads(c.value("chemistry_threads",8));
        int ac=1;char** av=argv;
        amrex::Initialize(ac,av,false);
        int result=0;
        try {
            amrex::ParmParse pg("geometry");pg.add("coord_sys",0);
            pg.addarr("prob_lo",amrex::Vector<double>(origin.begin(),origin.end()));
            pg.addarr("prob_hi",amrex::Vector<double>{origin[0]+length[0],origin[1]+length[1]});
            pg.addarr("is_periodic",amrex::Vector<int>(periodic.begin(),periodic.end()));
            amrex::ParmParse pa("amr");pa.addarr("n_cell",amrex::Vector<int>(nc.begin(),nc.end()));
            pa.add("max_level",levels);pa.add("ref_ratio",2);pa.add("blocking_factor",blocking);
            pa.add("max_grid_size",c.value("max_grid_size",32));pa.add("n_error_buf",c.value("tag_buffer",2));
            pa.add("grid_eff",c.value("grid_eff",0.7));pa.add("n_proper",2);pa.add("verbose",0);
            c["cfl"]=c.value("cfl",0.6);
            BlockAMR app(c,output);app.run();
        }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<'\n';result=1;}
        amrex::Finalize();return result;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
