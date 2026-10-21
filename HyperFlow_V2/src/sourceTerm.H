#ifndef SOURCE_TERM_H
#define SOURCE_TERM_H

#include <Eigen/Dense>

#include "preComputation.H"

/**
 * 统一源项写入 S_out（须已分配为 nVar×nface）。
 * 公共项：动量 pA·dlnA/dx + 摩擦；能量按 friction_energy_model
 * 选择绝热静止壁或历史摩擦功率损失模型。
 * isInjection=true 时叠加喷注质量/动量/能量/组分源。
 */
void sourceTerm_calculation(const Eigen::MatrixXd & pA,
                            const Eigen::MatrixXd & rhoA,
                            const Eigen::MatrixXd & u,
                            const InputData & input,
                            bool isInjection,
                            Eigen::MatrixXd & S_out);

#endif
