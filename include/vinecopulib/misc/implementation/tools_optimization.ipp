// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://vinecopulib.github.io/vinecopulib/.

#include <vinecopulib/misc/tools_stats.hpp>
#include <cppoptlib/boundedproblem.h>
#include <cppoptlib/solver/lbfgsbsolver.h>
#include <boost/math/tools/minima.hpp>
#include <iostream>

namespace vinecopulib {

//! Utilities for numerical optimization (based on Bobyqa)
namespace tools_optimization {

//! creates an Optimizer using the default controls, see CppNumericalSolvers.
//!
//! @param n_parameters Number of parameters to optimize
//! @param lower_bounds
//! @param upper_bounds
//! @param objective The optimizer's objective function
inline Optimizer::Optimizer(unsigned int n_parameters,
                            const Eigen::MatrixXd &lower_bounds,
                            const Eigen::MatrixXd &upper_bounds)
{
    if (n_parameters < 1) {
        throw std::runtime_error("n_parameters should be larger than 0.");
    }
    n_parameters_ = n_parameters;
    controls_ = cppoptlib::Criteria<double>::defaults();
    lb_ = lower_bounds;
    ub_ = upper_bounds;
}

//! set the optimizer's controls.
//!
//! @param interations Maximum number of iterations.
//! @param xDelta Minimum change in parameter vector.
//! @param fDelta Minimum change in cost function.
inline void Optimizer::set_controls(size_t iterations,
                                    double xDelta,
                                    double fDelta)
{
    controls_.iterations = iterations;
    controls_.xDelta = xDelta;
    controls_.fDelta = fDelta;
}

//! @name (Pseudo-) maximum likelihood estimation
//! @param f_data a pointer to a ParBicopOptData object.
//! @param n number of parameters.
//! @param x the parameters.
//! @{

//! evaluates the objective function for maximum likelihood estimation.
inline double mle_objective(void *f_data, long n, const double *x)
{
    ParBicopOptData *newdata = static_cast<ParBicopOptData *>(f_data);
    ++newdata->objective_calls;
    Eigen::Map<const Eigen::VectorXd> par(&x[0], n);
    newdata->bicop->set_parameters(par);
    return (-1) * newdata->bicop->pdf(newdata->U).array().log().sum();
}

//! evaluates the objective function for profile maximum likelihood
//! estimation.
inline double pmle_objective(void *f_data, long n, const double *x)
{
    ParBicopOptData *newdata = static_cast<ParBicopOptData *>(f_data);
    ++newdata->objective_calls;
    Eigen::VectorXd par = Eigen::VectorXd::Ones(n + 1);
    par(0) = newdata->par0;
    for (long i = 0; i < n; ++i) {
        par(i + 1) = x[i];
    }
    newdata->bicop->set_parameters(par);
    double nll = newdata->bicop->pdf(newdata->U).array().log().sum();
    nll *= -1;
    return nll;
}

//! @}

class mle_problem : public cppoptlib::BoundedProblem<double>
{
public:
    using Superclass = cppoptlib::BoundedProblem<double>;
    using typename Superclass::TVector;

    mle_problem(std::function<double(void *, long, const double *)> objective,
                long n_parameters,
                void *f_data) :
        Superclass(n_parameters), n_parameters_(n_parameters),
        objective_(objective), f_data_(f_data) {}

    double value(const TVector &x) {
        double *par = new double[n_parameters_];
        Eigen::VectorXd::Map(par, n_parameters_) = x.array();
        double res = objective_(f_data_, n_parameters_, par);
        delete[] par;
        return res;
    }

    void gradient(const TVector &x, TVector &grad) {
        Eigen::VectorXd parameters = x;
        double eps = 1e-3;
        double parscale = 1.0;

        //std::cout << "par:" << x << std::endl;
        for (size_t par = static_cast<size_t>(0); par < n_parameters_; ++par) {
            // get parameter +/- eps
            double par0 = parameters(par);
            double par1 = par0 + eps;
            double par2 = par0 - eps;

            // check bounds
            double ub = static_cast<double>(upperBound()(par));
            double lb = static_cast<double>(lowerBound()(par));
            double eps1 = eps;
            double eps2 = eps;
            if (par1 > ub) {
                par1 = ub;
                eps1 = par1 - par0;
            }
            if (par2 < lb) {
                par2 = lb;
                eps2 = par0 - par2;
            }

            // adjust for parscale
            par1 = par1 * parscale;
            par2 = par2 * parscale;

            // compute objective for both parameter values
            parameters(par) = par1;
            double result1 = value(parameters);
            parameters(par) = par2;
            double result2 = value(parameters);

            // compute gradient and restore initial parameter value
            grad(par) = (result1 - result2)/(eps1 + eps2);
            parameters(par) = par0;
            //std::cout << par << ", " << par0 << ", "  << result1 << ", "
            //          << result2 << ", "  << grad(par) << std::endl;
        }
    }

    long n_parameters_;
    std::function<double(void *, long, const double *)> objective_;
    void *f_data_;
};

//! solve the optimization problem.
//!
//! @param initial_parameters of starting values for the optimization
//!     algorithm.
//! @return the optimal parameters.
inline Eigen::VectorXd Optimizer::optimize(
    Eigen::VectorXd initial_parameters,
    std::function<double(void *, long, const double *)> objective,
    void *data)
{
    if (initial_parameters.size() != n_parameters_) {
        throw std::runtime_error("initial_parameters.size() should be n_parameters_.");
    }

    double eps = 1e-4;
    lb_ = lb_.array() + eps;
    ub_ = ub_.array() - eps;
    Eigen::VectorXd optimized_parameters = initial_parameters;
    if (n_parameters_ > 1) {
        // create problem and set bounds
        mle_problem problem(objective, n_parameters_, data);
        problem.setBoxConstraint(lb_, ub_);

        // create solver and set controls
        // maxiterations = 100, deltax = 1e-4, delta = 1e-4 * sample size
        cppoptlib::LbfgsbSolver<mle_problem> solver;
        double n = static_cast<double>((static_cast<ParBicopOptData *>(data))->U.rows());
        set_controls(100, eps, eps * n);
        solver.setStopCriteria(controls_);

        // optimize
        solver.minimize(problem, optimized_parameters);
    } else {
        auto f = [data, objective](double x) -> double {
            return objective(data, 1, &x);
        };
        auto result =
            boost::math::tools::brent_find_minima(f,
                                                  static_cast<double>(lb_(0)),
                                                  static_cast<double>(ub_(0)),
                                                  20);
        optimized_parameters(0) = result.first;
    }

    return optimized_parameters;
}
}

}
