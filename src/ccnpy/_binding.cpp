// pybind11 binding for the Classifier Chain Network C++ core.
//
// A thin shim that exposes the framework-agnostic C++ core (in ccncpp/) to
// NumPy. No algorithm lives here; Eigen <-> NumPy marshalling is handled by
// pybind11/eigen.h.

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "cc_network.h"
#include "cc_sequential.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core bindings for the Classifier Chain Network.";

    // Fit the full network. Returns (params, loss). X is feature-major
    // (m x n), Y is label-major (L x n), matching the core's contract.
    m.def(
        "ccn_fit",
        [](const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y,
           const Eigen::VectorXd& params, double q, double alpha, double c1,
           double c2, double tol, const std::string& loss_type,
           double heaviside_k, double heaviside_t) {
            CCNResult res = ccn_logistic(
                X, Y, params, q, alpha, c1, c2, tol, loss_type, heaviside_k,
                heaviside_t);
            return py::make_tuple(res.params, res.loss);
        },
        py::arg("X"), py::arg("Y"), py::arg("params"), py::arg("q"),
        py::arg("alpha"), py::arg("c1"), py::arg("c2"), py::arg("tol"),
        py::arg("loss_type"), py::arg("heaviside_k") = 0.0,
        py::arg("heaviside_t") = 0.0,
        "Fit the classifier chain network. Returns (params, loss).");

    // Fit a single sequential link (used for the informed initialization).
    m.def(
        "ccs_fit", &ccs_logistic,
        py::arg("X"), py::arg("y"), py::arg("alpha"), py::arg("c1"),
        py::arg("c2"), py::arg("tol"), py::arg("loss_type"),
        py::arg("heaviside_k") = 0.0, py::arg("heaviside_t") = 0.0,
        "Fit one sequential classifier-chain link. Returns the coefficients.");

    // Predicted probabilities. Returns an L x n matrix.
    m.def(
        "ccn_predict", &ccn_prediction,
        py::arg("X"), py::arg("params"), py::arg("L"),
        "Predicted probabilities (L x n) for feature-major X (m x n).");
}
