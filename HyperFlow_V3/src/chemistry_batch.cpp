#include "chemistry_batch.hpp"
#include <omp.h>
#include <cstdlib>
#include <exception>
#include <algorithm>

ChemistryBatch::ChemistryBatch(const std::string& mechanism, int threads) {
    if (threads<0) throw std::invalid_argument("chemistry_threads must be nonnegative");
    if (!threads) {
        threads=omp_get_max_threads();
        if (!std::getenv("OMP_NUM_THREADS")) threads=std::min(8,threads);
    }
    threads=std::max(1,std::min(threads,omp_get_thread_limit()));
    // Factories/YAML loading run serially; no phase, kinetics cache, or CVODE
    // workspace is shared with another worker or the flow/EOS phase.
    for (int i=0;i<threads;++i)
        workers.push_back(std::make_unique<Chemistry>(
            Cantera::newSolution(mechanism,"","none")));
}

void ChemistryBatch::advance(Field& field, double dt, size_t count) {
    if (omp_in_parallel()) throw std::logic_error("nested chemistry batch is unsupported");
    if (count>field.size() || !(dt>0) || !std::isfinite(dt))
        throw std::invalid_argument("invalid chemistry batch extent or interval");
    if (!count) { lastTeam=0; return; }
    Field trial(field.begin(),field.begin()+count);
    std::vector<std::exception_ptr> errors(count);
    const int team=std::min<size_t>(workers.size(),count);
    #pragma omp parallel num_threads(team)
    {
        #pragma omp single
        lastTeam=omp_get_num_threads();
        auto& chemistry=*workers[omp_get_thread_num()];
        #pragma omp for schedule(dynamic,8)
        for (size_t k=0;k<count;++k) {
            try { chemistry.advance(trial[k],dt); }
            catch (...) { errors[k]=std::current_exception(); }
        }
    }
    // Exceptions cannot cross an OpenMP region. Join first, then report the
    // lowest failing index. On failure the caller's entire field is unchanged.
    for (size_t k=0;k<count;++k) if (errors[k]) {
        try { std::rethrow_exception(errors[k]); }
        catch (const std::exception& e) {
            throw std::runtime_error("chemistry cell "+std::to_string(k)+": "+e.what());
        }
        catch (...) { throw std::runtime_error("unknown chemistry error at cell "+std::to_string(k)); }
    }
    std::copy(trial.begin(),trial.end(),field.begin());
}
