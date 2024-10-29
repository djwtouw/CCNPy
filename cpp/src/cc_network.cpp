#include <utility>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "Eigen/Eigen"
#include "cc_network.h"
#include "ccn_components.h"
#include "loss_components.h"
#include "loss_constants.h"


using namespace pybind11::literals;


struct CCNUpdateResults {
    CCNParams params;
    CCNGradient gradient;
    double loss = 0;
    double step_size = 0;
    bool update_found = false;
};


/**
 * @brief Compute inputs and results of the sigmoid activation function.
 *
 * Compute the input (z) and result of the sigmoid activation function (p) for
 * a single observation in the data.
 *
 * @param x The feature vector for observation i.
 * @param params The parameters of the classifier chain network.
 * @return A pair of two vectors containing the input and output for the
 * sigmoid activation function.
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd>
compute_z_and_p(Eigen::Ref<const Eigen::VectorXd> x, const CCNParams& params)
{
    // Start with x @ W + b for the contents of z
    Eigen::VectorXd z = params.b + params.W.transpose() * x;

    // Initialize p
    Eigen::VectorXd p(z.size());

    // Compute the predictions in P using an activation function (i.e. sigmoid)
    // and add its prediction to the subsequent labels
    for (int l1 = 0; l1 < params.L; l1++) {
        p(l1) = sigmoid(z(l1));

        for (int l2 = l1 + 1; l2 < params.L; l2++) {
            // The parameter that governs the effect of the prediction for
            // label 1 on the prediction for label 2
            double c_l2l1 = params.c_lk(l2, l1);

            z(l2) += c_l2l1 * p(l1);
        }
    }

    return std::make_pair(z, p);
}


/**
 * @brief Compute the value and gradient of the loss function for a single
 * observation in the data.
 *
 * Compute the contribution of a single observation to the loss and the
 * gradient of the parameters of the model. Potential penalty terms applied to
 * the parameters are not included.
 *
 * @param x The feature vector for observation i.
 * @param y The outcome for observation i (0 or 1).
 * @param params The parameters of the classifier chain network.
 * @param q The order of the norm applied to the vector of label-specific
 * losses.
 * @param lc Information about the loss function applied to the predictions.
 * @return A pair containing the value of the loss function and the gradient
 * for a single observation.
 */
std::pair<double, CCNGradient>
ccn_loss_gradient_i(Eigen::Ref<const Eigen::VectorXd> x,
                    Eigen::Ref<const Eigen::VectorXd> y,
                    const CCNParams& params, double q,
                    const LossConstants &lc)
{
    // Initialize gradient
    CCNGradient gradient(params.m, params.L);

    // Compute z and p
    auto [z, p] = compute_z_and_p(x, params);

    // Compute h
    Eigen::VectorXd h(p.size());
    for (int i = 0; i < params.L; i++) {
        if (lc.loss_type == "cross_entropy") {
                h(i) = cross_entropy(p(i), y(i));
        } else if (lc.loss_type == "heaviside") {
                h(i) = heaviside(p(i), y(i), lc.heaviside_k, lc.heaviside_t);
        }
    }

    // Compute lq-norm
    double lq = q_norm(h, q);

    // Compute gradient, start with the last label
    int l1 = params.L - 1;

    while (l1 >= 0) {
        // Gradient of lq-norm w.r.t. H (the output of the cross entropy loss)
        double gradient_lq = q_norm_grad(h(l1), lq, q);

        // Gradient of the specified loss w.r.t. P
        double gradient_h = 0.0;
        if (lc.loss_type == "cross_entropy") {
            gradient_h = cross_entropy_grad(p(l1), y(l1));
        } else if (lc.loss_type == "heaviside") {
            gradient_h = heaviside_grad(h(l1), y(l1), lc.heaviside_k);
        }

        // Gradient of cross entropy w.r.t. P
        double gradient_lq_h = gradient_lq * gradient_h;

        // Gradient of the sigmoid activation function
        double gradient_sig = sigmoid_grad(p(l1));

        // Gradient for the bias
        gradient.b[l1] += gradient_lq_h;
        gradient.b[l1] *= gradient_sig;

        // Gradient for the weights
        gradient.W.col(l1) += gradient_lq_h * x;
        gradient.W.col(l1) *= gradient_sig;

        for (int l2 = 0; l2 < l1; l2++) {
            // Get the value for c that governs the effect of label l2 on label
            // l1
            double c_l1l2 = params.c_lk(l1, l2);

            // Add to the gradient of the biases of the previous labels
            gradient.b(l2) += c_l1l2 * gradient.b(l1);

            // Add to the gradient of the weights of the previous labels
            gradient.W.col(l2) += c_l1l2 * gradient.W.col(l1);
        }

        // The part for c
        for (int l2 = 0; l2 < l1; l2++) {
            int c_idx_l1l2 = (l1 * l1 - l1) / 2 + l2;

            // Add the cross entropy part to the result
            gradient.c(c_idx_l1l2) += gradient_lq_h * p(l2);

            // Multiply by the sigmoid part
            gradient.c(c_idx_l1l2) *= gradient_sig;

            for (int l3 = l2 + 1; l3 < l1; l3++) {
                // Get the value for c that governs the effect of label l3 on
                // label l1
                double c_l1l3 = params.c_lk(l1, l3);

                // Add to the gradient of the c of the previous labels. To
                // fully understand the way the indices work, check the formula
                // in the paper
                int c_idx_l3l2 = (l3 * l3 - l3) / 2 + l2;
                gradient.c(c_idx_l3l2) += c_l1l3 * gradient.c(c_idx_l1l2);
            }
        }

        // Decrement l1 by one
        l1 -= 1;
    }

    return std::make_pair(lq, gradient);
}


/**
 * @brief Compute the value and gradient of the loss function.
 *
 * The contribution of each observation to the loss and gradient is computed.
 * After this, the penalty term is added to the sum of the individual losses
 * and gradients.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param Y The outcome matrix.
 * @param params The parameters of the classifier chain network.
 * @param q The order of the norm applied to the vector of label-specific
 * losses.
 * @param alpha The regularization parameter.
 * @param lc Information about the loss function applied to the predictions.
 * @return A pair containing the value of the loss function and the gradient.
 */
std::pair<double, CCNGradient>
ccn_loss_gradient(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                  const CCNParams& params, double q, double alpha,
                  const LossConstants &lc)
{
    int n = int(X.cols());
    int L = params.L;
    int m = params.m;

    // Initialize gradient
    CCNGradient gradient(params.m, params.L);

    // Initialize the loss
    double loss = 0;

    // Iterate over all observations in the data
    for (int i = 0; i < n; i++) {
        // Compute gradient and loss for observation i
        auto [loss_i, gradient_i] =
                ccn_loss_gradient_i(X.col(i), Y.col(i), params, q, lc);

        // Add gradient and loss for observation i
        gradient.add(gradient_i);
        loss += loss_i;
    }

    // Compute the scale factor for the first part of the loss function
    double scale_factor = 1.0 / (n * std::pow(double(L), 1.0 / q));

    // Scale the loss
    loss *= scale_factor;

    // Scale the gradient
    gradient.multiply(scale_factor);

    // Add regularization
    loss += alpha * params.squaredNorm() / (params.size - L);

    // Add the gradient to the regularization part
    CCNGradient gradient_reg(m, L);
    gradient_reg.W = 2 * alpha * params.W / (params.size - L);
    gradient_reg.c = 2 * alpha * params.c / (params.size - L);
    gradient.add(gradient_reg);

    return std::make_pair(loss, gradient);
}


/**
 * @brief Compute the update for the parameters using the Wolfe conditions.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param Y The outcome matrix.
 * @param params0 The parameters of the classifier chain network.
 * @param q The order of the norm applied to the vector of label-specific
 * losses.
 * @param alpha The regularization parameter.
 * @param gradient0 The current gradient.
 * @param descent_direction The current descent direction.
 * @param loss0 The current value of the loss function.
 * @param c1 Wolfe condition parameter 1 (descent).
 * @param c2 Wolfe condition parameter 2 (curvature).
 * @param lc Information about the loss function applied to the predictions.
 * @return A struct holding the variables relevant to the update.
 */
CCNUpdateResults
ccn_update_via_wolfe(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
                     const CCNParams& params0, double q, double alpha,
                     const CCNGradient& gradient0,
                     const CCNDescentDirection& descent_direction,
                     double loss0, double c1, double c2,
                     const LossConstants &lc)
{
    // Set initial step size and its min and max
    double step_size = 1;
    double step_size_min = 0;
    double step_size_max = -1;

    // Variable to track whether an appropriate step size was found
    bool update_found = false;

    // Use a variable to store the largest step size that is found for which a
    // sufficient decrease of the loss function
    // is observed
    double step_size_backup = -1;

    // New parameter values
    CCNParams params1 = params0;

    // Constants for Wolfe conditions
    double wolfe_temp = gradient0.dot(descent_direction);
    double wolfe_1 = c1 * wolfe_temp;
    double wolfe_2 = c2 * wolfe_temp;

    // Definition of loss and gradient
    double loss1 = 0;
    CCNGradient gradient1(params0.m, params0.L);

    for (int i = 0; i < 20; i ++) {
        // Compute updated parameters, loss, and gradient
        params1.update(descent_direction, step_size);
        std::tie(loss1, gradient1) =
                ccn_loss_gradient(X, Y, params1, q, alpha, lc);

        // Check if the loss did not decrease enough
        if (loss1 > loss0 + step_size * wolfe_1) {
            // If so, reduce the maximum step size to the current step size and
            // adjust the current step size to the midpoint between the minimum
            // and maximum step sizes
            step_size_max = step_size;
            step_size = 0.5 * (step_size_min + step_size_max);
        }
        // Check if the slope has not been reduced enough
        else if (gradient1.dot(descent_direction) < wolfe_2) {
            // If this condition is evaluated, that means that the current step
            // size satisfied the first Wolfe condition but not the second.
            // Hence, store this value, so it can be used if no step size
            // satisfying the second Wolfe condition can be found within the
            // given maximum number of iterations
            if (step_size > step_size_backup) {
                step_size_backup = step_size;
            }

            // If the check is true, increase the minimum step size to the
            // current step size and adjust the current step size
            step_size_min = step_size;

            // If the maximum step size is not infinite, indicated by it being
            // larger than the minimum
            if (step_size_max > step_size_min) {
                // Set the step size to the midpoint between the minimum and
                // maximum step sizes
                step_size = 0.5 * (step_size_min + step_size_max);
            } else {
                // Otherwise, set the step size to twice the minimum
                step_size = 2 * step_size_min;
            }
        }
        // If both checks failed, the Wolfe conditions are met and the loop can
        // be broken
        else {
            update_found = true;
            break;
        }

        // Revert the changes to the parameters
        params1 = params0;
    }

    // Construct a container for the results
    CCNUpdateResults result;

    if (update_found) {
        result.params = params1;
        result.gradient = gradient1;
        result.loss = loss1;
        result.step_size = step_size;
        result.update_found = update_found;
    } else {
        result.step_size = step_size_backup;
    }

    return result;
}


/**
 * @brief Update the inverse of the Hessian.
 *
 * @param H The current inverse of the Hessian.
 * @param gradient0 The previous gradient.
 * @param gradient1 The new gradient.
 * @param step_size The step size satisfying the Wolfe conditions.
 * @param descent_direction The descent direction used for the update of the
 * parameters.
 */
void
ccn_update_hessian_inverse(Eigen::MatrixXd& H,
                           const Eigen::VectorXd& gradient0,
                           const Eigen::VectorXd& gradient1, double step_size,
                           const Eigen::VectorXd& descent_direction)
{
    // Compute the step that is taken
    Eigen::VectorXd s = step_size * descent_direction;

    // Compute the difference in gradients
    Eigen::VectorXd y = gradient1 - gradient0;

    // Compute rho
    double rho = 1 / s.dot(y);

    // Compute intermediate terms
    Eigen::VectorXd Hy = H * y;
    Eigen::MatrixXd outer_ss = s * s.transpose();
    Eigen::MatrixXd temp1 = (1.0 / rho + y.dot(Hy)) * outer_ss;
    Eigen::MatrixXd temp2 = Hy * s.transpose();

    // Update the inverse of the Hessian
    H += (rho * rho) * temp1 - rho * (temp2 + temp2.transpose());
}


/**
 * @brief Minimize the loss associated with the classifier chain network.
 *
 * @param X X The data matrix, where each column holds the features of a single
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
pybind11::dict
ccn_logistic(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
             const Eigen::VectorXd& params_vec, double q, double alpha,
             double c1, double c2, double tol, std::string loss_type,
             double heaviside_k, double heaviside_t)
{
    // Preliminaries
    int L = int(Y.rows());
    int m = int(X.rows());

    // Initialize the parameters
    CCNParams params1(params_vec, m, L);

    // Initialize loss information struct
    LossConstants lc;
    lc.loss_type = loss_type;
    lc.heaviside_k = heaviside_k;
    lc.heaviside_t = heaviside_t;

    // Compute the first loss and gradient
    auto [loss1, gradient1] = ccn_loss_gradient(X, Y, params1, q, alpha, lc);

    // Set the value of the previous loss to something sufficiently large
    double loss0 = 2 * loss1;

    // Initial estimate for the inverse of the Hessian
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(params1.size, params1.size);

    while (loss0 / loss1 - 1 > tol) {
        // Update the value of loss0, as computations later in the loop will
        // overwrite loss1
        loss0 = loss1;

        // Compute the descent direction
        CCNDescentDirection descent_direction(gradient1, H);

        // Compute the step size and, corresponding to the updated parameters,
        // the updated gradient and loss
        CCNUpdateResults wolfe_update =
                ccn_update_via_wolfe(X, Y, params1, q, alpha, gradient1,
                                     descent_direction, loss1, c1, c2, lc);

        if (wolfe_update.update_found) {
            // Update the estimate of the inverse of the Hessian
            ccn_update_hessian_inverse(H, gradient1.flatten(),
                                       wolfe_update.gradient.flatten(),
                                       wolfe_update.step_size,
                                       descent_direction.flattened);

            params1 = wolfe_update.params;
            gradient1 = wolfe_update.gradient;
            loss1 = wolfe_update.loss;
        } else {
            // If no update that satisfies the Wolfe conditions was found, the
            // only relevant part of the output is the step size. If the step
            // size is negative, no update was found that satisfies the
            // sufficient decrease condition. In that case, the minimization
            // can be terminated
            if (wolfe_update.step_size <= 0) {
                break;
            }

            // Compute parameters, loss, and gradient and set H to the identity
            // matrix again
            params1.update(descent_direction, wolfe_update.step_size);
            std::tie(loss1, gradient1) =
                    ccn_loss_gradient(X, Y, params1, q, alpha, lc);
            H = Eigen::MatrixXd::Identity(params1.size, params1.size);
        }

        /*pybind11::print(
            "loss = {:.5f} | step size = {:.5f}"_s.format(loss1, wolfe_update.step_size)
        );*/
    }

    // Construct the result
    pybind11::dict result;
    result["params"] = params1.flatten();
    result["loss"] = loss1;

    return result;
}
