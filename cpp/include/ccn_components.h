#ifndef CCN_COMPONENTS_H
#define CCN_COMPONENTS_H

#include <cmath>
#include <utility>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


struct CCNGradient;


struct CCNDescentDirection {
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::MatrixXd W;
    Eigen::VectorXd flattened;

    /**
     * @brief Parameterized constructor for CCNDescentDirection.
     *
     * Initialize a CCNDescentDirection based on the provided gradient and
     * inverse of the Hessian.
     *
     * @param gradient The gradient used to compute the descent direction.
     * @param hessian_inv The inverse of the Hessian.
     */
    CCNDescentDirection(const CCNGradient& gradient,
                        const Eigen::MatrixXd& hessian_inv);
};


struct CCNGradient {
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::MatrixXd W;
    int m = 0;
    int L = 0;
    int size = 0;


    /**
     * @brief Default constructor for CCNGradient.
     */
    CCNGradient() = default;

    /**
     * @brief Parameterized constructor for CCNGradient.
     *
     * @param m The number of features in the data.
     * @param L The number of labels in the data.
     */
    CCNGradient(int m, int L) : m(m), L(L)
    {
        b = Eigen::VectorXd::Zero(L);
        W = Eigen::MatrixXd::Zero(m, L);
        c = Eigen::VectorXd::Zero((L * L - L) / 2);
        size = L + m * L + (L * L - L) / 2;
    }

    /**
     * @brief Add another gradient to this gradient.
     *
     * @param other The other gradient.
     */
    void add(const CCNGradient& other)
    {
        b += other.b;
        W += other.W;
        c += other.c;
    }

    /**
     * @brief Multiply the gradient by a scalar value.
     *
     * @param value The value the gradient is multiplied with.
     */
    void multiply(double value)
    {
        b *= value;
        W *= value;
        c *= value;
    }

    /**
     * @brief Compute the dot product of the gradient with the descent
     * direction.
     *
     * @param descent_direction The descent direction.
     * @return The value of the dot product.
     */
    double dot(const CCNDescentDirection& descent_direction) const
    {
        double result = b.dot(descent_direction.b);
        result += W.cwiseProduct(descent_direction.W).sum();
        result += c.dot(descent_direction.c);

        return result;
    }

    /**
     * @brief Flatten the parameter struct into a single vector.
     *
     * @return A one-dimensional vector holding the gradient.
     */
    Eigen::VectorXd flatten() const
    {
        // Initialize result
        Eigen::VectorXd result(size);

        // Fill in b
        result.head(b.size()) = b;

        // Fill in W
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < m; j++) {
                int index = L + i * m + j;
                result(index) = W(j, i);
            }
        }

        // Fill in c
        result.tail(c.size()) = c;

        return result;
    }
};


struct CCNParams {
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::MatrixXd W;
    int m = 0;
    int L = 0;
    int size = 0;

    /**
     * @brief Default constructor for CCNParams.
     */
    CCNParams() = default;

    /**
     * @brief Parameterized constructor for CCNParams.
     *
     * Constructor that uses the number of features and labels in the data to
     * construct the container holding the parameters.
     *
     * @param m The number of features in the data.
     * @param L The number of labels in the data.
     */
    CCNParams(int m, int L) : m(m), L(L)
    {
        b = Eigen::VectorXd(L);
        W = Eigen::MatrixXd(m,L);
        c = Eigen::VectorXd((L * L - L) / 2);
        size = L + m * L + (L * L - L) / 2;
    }

    /**
     * @brief Parameterized constructor for CCNParams.
     *
     * Constructor that uses an existing parameter vector to construct the
     * container holding the parameters.
     *
     * @param params The parameter vector.
     * @param m The number of features in the data.
     * @param L The number of labels in the data.
     */
    CCNParams(const Eigen::VectorXd& params, int m, int L) : m(m), L(L)
    {
        b = Eigen::VectorXd(L);
        W = Eigen::MatrixXd(m,L);
        c = Eigen::VectorXd((L * L - L) / 2);
        size = L + m * L + (L * L - L) / 2;

        // Fill in b and c
        b = params.head(L);
        c = params.tail((L * L - L) / 2);

        // Fill in W
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < m; j++) {
                int index = L + i * m + j;
                W(j, i) = params(index);
            }
        }
    }

    /**
     * @brief Return the effect of label k on label l.
     *
     * Label k must be smaller than label l.
     *
     * @param l The index of the label that is affected by label k.
     * @param k The index of the label that affects label l.
     * @return The value of the effect that label k has on label l according to
     * the parameters of the classifier chain network.
     */
    double c_lk(int l, int k) const
    {
        // Compute index
        int index = (l * l - l) / 2 + k;

        return c(index);
    }

    /**
     * @brief The sum of squares of the parameters in the model, excluding the
     * biases.
     *
     * @return The sum of squares of the parameters.
     */
    double squaredNorm() const
    {
        // Compute the squared norm of all parameter containers excluding the
        // biases and add them
        double result = W.squaredNorm();
        result += c.squaredNorm();

        return result;
    }

    /**
     * @brief Update the parameters using the descent direction and step size.
     *
     * @param descent_direction The descent direction.
     * @param step_size The size of the step taken into the descent direction.
     */
    void update(const CCNDescentDirection& descent_direction, double step_size)
    {
        b += step_size * descent_direction.b;
        W += step_size * descent_direction.W;
        c += step_size * descent_direction.c;
    }

    /**
     * @brief Flatten the parameter struct into a single vector.
     *
     * @return A one-dimensional vector holding the parameters.
     */
    Eigen::VectorXd flatten() const
    {
        // Initialize result
        Eigen::VectorXd result(size);

        // Fill in b
        result.head(b.size()) = b;

        // Fill in W
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < m; j++) {
                int index = L + i * m + j;
                result(index) = W(j, i);
            }
        }

        // Fill in c
        result.tail(c.size()) = c;

        return result;
    }
};

#endif //CCN_COMPONENTS_H
