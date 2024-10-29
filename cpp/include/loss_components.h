#ifndef LOSS_COMPONENTS_H
#define LOSS_COMPONENTS_H

#include <cmath>
#include "Eigen/Eigen"


/**
 * @brief Compute the sigmoid activation function value.
 *
 * @param z The input value.
 * @return The value of the sigmoid activation function for the input.
 */
double sigmoid(double z);


/**
 * @brief Compute the gradient of the sigmoid function based on its output.
 *
 * The gradient of the sigmoid activation function can easily be expressed as a
 * function of its output. If p is the output, the gradient is p * (p - 1).
 *
 * @param p The output of the sigmoid activation function.
 * @return The value of the gradient of the sigmoid activation function.
 */
double sigmoid_grad(double p);


/**
 * @brief Compute the binary cross entropy loss for a probability p given the
 * true outcome y.
 *
 * @param p The (predicted) probability that a 1 occurs.
 * @param y The true value (0 or 1).
 * @return The value of the binary cross entropy loss.
 */
double cross_entropy(double p, double y);


/**
 * @brief Compute the gradient of the binary cross entropy loss with respect to
 * the probability p given the true outcome y.
 *
 * @param p The (predicted) probability that a 1 occurs.
 * @param y The true value (0 or 1).
 * @return The value of the gradient of the binary cross entropy loss with
 * respect to p.
 */
double cross_entropy_grad(double p, double y);


/**
 *@brief Compute the heaviside stepfunction loss with respect to
 * the probability p given the true outcome y.
 *
 * @param p The (predicted) probability that a 1 occurs.
 * @param y The true value (0 or 1).
 * @param k The parameter governing the steepness of the heaviside stepfunction.
 * @param t The threshold parameter, for p > t -> heaviside(p) > 0.5.
 * @returns The value of the heaviside stepfunction loss
 */
double heaviside(double p, double y, double k, double t);


/**
 *@brief Compute the heaviside stepfunction loss with respect to
 * the probability p given the true outcome y.
 *
 * @param hsl The value of the heaviside stepfunction loss.
 * @param y The true value (0 or 1).
 * @param k The parameter governing the steepness of the heaviside stepfunction.
 * @returns The value of the gradient of the heaviside stepfunction loss with
 * respect to p.
 */
double heaviside_grad(double hsl, double y, double k);


/**
 * @brief Compute the q-norm.
 *
 * The vector h contains loss values associated with multiple predictions, it
 * is essential that these values are nonnegative and that lower is considered
 * better.
 *
 * @param h The vector containing the losses assoicated with the individual
 * predictions.
 * @param q The order of the norm.
 * @return The value of the q-norm of the vector h.
 */
double q_norm(const Eigen::VectorXd& h, double q);


/**
 * @brief Compute the gradient of the q-norm with respect to one of the
 * elements of the vector used to compute the q-norm.
 *
 * Given the value of the q-norm of the original vector, compute the gradient
 * provided that one of its elements has the value h.
 *
 * @param h The value of the element for which the gradient is computed.
 * @param lq The value of the q-norm of the original vector.
 * @param q The order of the norm......
 * @return
 */
double q_norm_grad(double h, double lq, double q);


#endif //LOSS_COMPONENTS_H
