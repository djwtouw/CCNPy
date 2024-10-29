#include <utility>
#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <string>

#include "Eigen/Eigen"
#include "loss_components.h"
#include "loss_constants.h"


using namespace pybind11::literals;


struct UpdateResults {
    Eigen::VectorXd params;
    Eigen::VectorXd gradient;
    double loss = 0;
    double step_size = 0;
    bool update_found = false;
};


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
 * @param params The parameter vector.
 * @param lc Information about the loss function applied to the predictions.
 * @return A pair containing the value of the loss function and the gradient
 * for a single observation.
 */
std::pair<double, Eigen::VectorXd>
ccs_loss_gradient_i(Eigen::Ref<const Eigen::VectorXd> x, double y,
                    const Eigen::VectorXd &params, const LossConstants &lc)
{
    // Initialize vector holding the gradient
    Eigen::VectorXd gradient(params.size());

    // Compute components of the loss function
    double z = params(0) + x.dot(params.tail(x.size()));
    double p = sigmoid(z);

    // Compute specified loss on the probability p
    double h = 0.0;
    if (lc.loss_type == "cross_entropy") {
        h = cross_entropy(p, y);
    } else if (lc.loss_type == "heaviside") {
        h = heaviside(p, y, lc.heaviside_k, lc.heaviside_t);
    }

    // Compute partial gradient of the specified loss on the probability p
    double gradient_h = 0.0;
    if (lc.loss_type == "cross_entropy") {
        gradient_h = cross_entropy_grad(p, y);
    } else if (lc.loss_type == "heaviside") {
        gradient_h = heaviside_grad(h, y, lc.heaviside_k);
    }

    // Compute partial gradient of the sigmoid function
    double gradient_sig = sigmoid_grad(p);

    // Construct gradient
    gradient(0) = gradient_h * gradient_sig;
    gradient.tail(x.size()) = gradient(0) * x;

    return std::make_pair(h, gradient);
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
 * @param y The outcome vector.
 * @param params The parameter vector.
 * @param alpha The regularization parameter.
 * @param lc Information about the loss function applied to the predictions.
 * @return A pair containing the value of the loss function and the gradient.
 */
std::pair<double, Eigen::VectorXd>
ccs_loss_gradient(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                  const Eigen::VectorXd& params, double alpha,
                  const LossConstants &lc)
{
    // Preliminaries
    int n = int(X.cols());

    // Initialize gradient
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(params.size());

    // Initialize the loss
    double loss = 0;

    // Iterate over all observations in the data
    for (int i = 0; i < n; i++) {
        // Compute gradient and loss for observation i
        auto [loss_i, gradient_i] =
                ccs_loss_gradient_i(X.col(i), y(i), params, lc);

        // Add gradient and loss for observation i
        gradient += gradient_i;
        loss += loss_i;
    }

    // Scale factor for the first term of the loss function
    double scale_factor = 1.0 / n;

    // Scale the loss
    loss *= scale_factor;

    // Scale the gradient
    gradient *= scale_factor;

    // The number of parameters excluding the intercept
    int m = int(X.rows());

    // Add regularization to the loss
    loss += alpha * params.tail(m).squaredNorm() / m;

    // Add the gradient of the regularization part
    gradient.tail(m) += 2 * alpha * params.tail(m) / m;

    return std::make_pair(loss, gradient);
}


/**
 * @brief Compute the update for the parameter vector using the Wolfe
 * conditions.
 *
 * @param X The data matrix, where each column holds the features of a single
 * observation.
 * @param y The outcome vector.
 * @param params0 The parameter vector.
 * @param alpha The regularization parameter.
 * @param gradient0 The current gradient.
 * @param descent_direction The current descent direction.
 * @param loss0 The current value of the loss function.
 * @param c1 Wolfe condition parameter 1 (descent).
 * @param c2 Wolfe condition parameter 2 (curvature).
 * @param lc Information about the loss function applied to the predictions.
 * @return A struct holding the variables relevant to the update.
 */
UpdateResults
ccs_update_via_wolfe(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                     const Eigen::VectorXd& params0, double alpha,
                     const Eigen::VectorXd& gradient0,
                     const Eigen::VectorXd& descent_direction, double loss0,
                     double c1, double c2, const LossConstants &lc)
{
    // Set initial step size and its min and max
    double step_size = 1;
    double step_size_min = 0;
    double step_size_max = -1;

    // Variable to track whether an appropriate step size was found
    bool update_found = false;

    // Use a variable to store the largest step size that is found for which a
    // sufficient decrease of the loss function is observed
    double step_size_backup = -1;

    // New parameter values
    Eigen::VectorXd params1(params0);

    // Constants for Wolfe conditions
    double wolfe_temp = gradient0.dot(descent_direction);
    double wolfe_1 = c1 * wolfe_temp;
    double wolfe_2 = c2 * wolfe_temp;

    // Definition of loss and gradient
    double loss1;
    Eigen::VectorXd gradient1(params0.size());

    for (int i = 0; i < 20; i ++) {
        // Compute updated parameters, loss, and gradient
        params1 += descent_direction * step_size;
        std::tie(loss1, gradient1) = ccs_loss_gradient(X, y, params1, alpha, lc);

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
            // larger than the minimum...
            if (step_size_max > step_size_min) {
                // ... set the step size to the midpoint between the minimum
                // and maximum step sizes
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
    UpdateResults result;

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
ccs_update_hessian_inverse(Eigen::MatrixXd& H,
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
             double heaviside_k, double heaviside_t)
{
    // Initialize the parameters
    Eigen::VectorXd params1 = Eigen::VectorXd::Zero(1 + X.rows());

    // Initialize loss information struct
    LossConstants lc;
    lc.loss_type = loss_type;
    lc.heaviside_k = heaviside_k;
    lc.heaviside_t = heaviside_t;

    // Compute the first loss and gradient
    auto [loss1, gradient1] = ccs_loss_gradient(X, y, params1, alpha, lc);

    // Set the value of the previous loss to something sufficiently large
    double loss0 = 2 * loss1;

    // Initial estimate for the inverse of the Hessian
    Eigen::MatrixXd H =
            Eigen::MatrixXd::Identity(params1.size(), params1.size());

    while (loss0 / loss1 - 1 > tol) {
        // Update the value of loss0, as computations later in the loop will
        // overwrite loss1
        loss0 = loss1;

        // Compute the descent direction
        Eigen::VectorXd descent_direction = -H * gradient1;

        // Compute the step size and, corresponding to the updated parameters,
        // the updated gradient and loss
        UpdateResults wolfe_update =
                ccs_update_via_wolfe(X, y, params1, alpha, gradient1,
                                     descent_direction, loss1, c1, c2, lc);

        if (wolfe_update.update_found) {
            // Update the estimate of the inverse of the Hessian
            ccs_update_hessian_inverse(H, gradient1, wolfe_update.gradient,
                                       wolfe_update.step_size,
                                       descent_direction);

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
            params1 += wolfe_update.step_size * descent_direction;
            std::tie(loss1, gradient1) =
                    ccs_loss_gradient(X, y, params1, alpha, lc);
            H = Eigen::MatrixXd::Identity(params1.size(), params1.size());
        }

        /*pybind11::print(
            "loss = {:.5f} | step size = {:.5f}"_s.format(loss1, wolfe_update.step_size)
        );*/
    }

    return params1;
}
