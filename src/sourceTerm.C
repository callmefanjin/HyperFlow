#include "sourceTerm.H"

#include <Eigen/Dense>

using namespace Eigen;

MatrixXd sourceTerm_process1(MatrixXd & pA,
                             MatrixXd & /*rhoA*/,
                             MatrixXd & /*u*/,
                             const InputData & input)
{
    MatrixXd sourceTerm = MatrixXd::Zero(input.nVar, input.nface);
    // Momentum: pA · dlnA/dx
    sourceTerm.block(1, 0, 1, input.nface) = pA.array() * input.dlnA_dx.array();
    return sourceTerm;
}
