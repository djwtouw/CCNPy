#ifndef CC_NETWORK_H
#define CC_NETWORK_H

#include <Eigen/Dense>


struct CCNResult {
    Eigen::VectorXd params;
    double loss;
};


/**
 * @brief Minimize the loss associated with the classifier chain network.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param Y The outcome matrix.
 * @param params_vec The initial estimate of the model parameters.
 * @param q The order of the norm applied to the vector of label-specific
 * losses.
 * @param alpha The regularization parameter.
 * @param c1 Wolfe condition parameter 1 (descent).
 * @param c2 Wolfe condition parameter 2 (curvature).
 * @param tol The tolerance to determine convergence.
 * @param loss_type The type of loss applied to the predictions.
 * @param heaviside_k The parameter governing the steepness of the heaviside
 * stepfunction.
 * @param heaviside_t The threshold parameter, for p > t -> heaviside(p) > 0.5.
 * @return The optimization results.
 */
CCNResult
ccn_logistic(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
             const Eigen::VectorXd& params_vec, double q, double alpha,
             double c1, double c2, double tol, std::string loss_type,
             double heaviside_k, double heaviside_t);

/**
 * @brief Minimize the loss associated with the classifier chain network.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param params_vec The initial estimate of the model parameters.
 * @param L The number of labels.
 * @return The prediction.
 */
Eigen::MatrixXd
ccn_prediction(const Eigen::MatrixXd& X, const Eigen::VectorXd& params_vec,
               int L);

#endif //CC_NETWORK_H
