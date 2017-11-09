// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://vinecopulib.github.io/vinecopulib/.

#pragma once

#include <vinecopulib/misc/tools_bobyqa.hpp>
#include <vinecopulib/misc/tools_eigen.hpp>
#include <vinecopulib/bicop/parametric.hpp>
#include <cppoptlib/meta.h>

namespace vinecopulib {

namespace tools_optimization {

//! @brief A helper struct for (profile) maximum likelihood estimation
typedef struct
{
    const Eigen::Matrix<double, Eigen::Dynamic, 2> &U; //!< the data.
    vinecopulib::ParBicop *bicop; //!< a pointer to the bivariate copula to optimize.
    double par0;  //!< main dependence parameter.
    unsigned int objective_calls; //!< number of evaluations of the objective.
} ParBicopOptData;


//! @brief A class for optimization (wrapping CppNumericalSolvers).
class Optimizer
{
public:
    Optimizer(unsigned int n_parameters,
              const Eigen::MatrixXd &lower_bounds,
              const Eigen::MatrixXd &upper_bounds);

    void set_controls(size_t iterations,
                      double xDelta,
                      double fDelta);

    Eigen::VectorXd optimize(Eigen::VectorXd initial_parameters,
                             std::function<double(void *, long,
                                                  const double *)> objective,
                             void *f_data);

private:
    unsigned int n_parameters_;
    cppoptlib::Criteria<double> controls_;
    Eigen::MatrixXd lb_;
    Eigen::MatrixXd ub_;
};
}

}

#include <vinecopulib/misc/implementation/tools_optimization.ipp>
