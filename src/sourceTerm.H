#ifndef SOURCE_TERM_H
#define SOURCE_TERM_H

#include <Eigen/Dense>

#include "preComputation.H"

/** Flow-substep source: geometric momentum term pA·dlnA/dx only */
Eigen::MatrixXd sourceTerm_process1(Eigen::MatrixXd & pA,
                                    Eigen::MatrixXd & rhoA,
                                    Eigen::MatrixXd & u,
                                    const InputData & input);

#endif
