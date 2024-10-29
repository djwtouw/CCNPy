#ifndef SEQUENTIAL_CC_H
#define SEQUENTIAL_CC_H

#include "Eigen/Eigen"

/**
 * @brief Minimize the loss associated with the logistic regression.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param y The outcome vector.
 * @param alpha The regularization parameter.
 * @param c1 Wolfe condition parameter 1 (descent).
 * @param c2 Wolfe condition parameter 2 (curvature).
 * @param tol The tolerance to determine convergence.
 * @param loss_type The type of loss applied to the predictions.
 * @param heaviside_k The parameter governing the steepness of the heaviside
 * stepfunction.
 * @param heaviside_t The threshold parameter, for p > t -> heaviside(p) > 0.5.
 * @return The optimized parameter vector.
 */
Eigen::VectorXd
ccs_logistic(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double alpha,
             double c1, double c2, double tol, std::string loss_type,
             double heaviside_k, double heaviside_t);

#endif //SEQUENTIAL_CC_H
