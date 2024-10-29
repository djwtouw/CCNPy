#include "Eigen/Eigen"
#include "ccn_components.h"


/**
 * @brief Parameterized constructor for CCNDescentDirection.
 *
 * Initialize a CCNDescentDirection based on the provided gradient and inverse
 * of the Hessian.
 *
 * @param gradient The gradient used to compute the descent direction.
 * @param hessian_inv The inverse of the Hessian.
 */
CCNDescentDirection::CCNDescentDirection(const CCNGradient& gradient,
                                         const Eigen::MatrixXd& hessian_inv)
{
    b = Eigen::VectorXd(gradient.L);
    W = Eigen::MatrixXd(gradient.m, gradient.L);
    c = Eigen::VectorXd(gradient.c.size());

    // Compute the descent direction
    flattened = -hessian_inv * gradient.flatten();

    // Fill in b
    b = flattened.head(b.size());

    // Fill in W
    for (int i = 0; i < gradient.L; i++) {
        for (int j = 0; j < gradient.m; j++) {
            int index = gradient.L + i * gradient.m + j;
            W(j, i) = flattened(index);
        }
    }

    // Fill in c
    c = flattened.tail(c.size());
}
