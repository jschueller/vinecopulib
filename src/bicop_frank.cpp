// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://tvatter.github.io/vinecopulib/.

#include "bicop_frank.hpp"
#include "tools_c.h"

// constructor
FrankBicop::FrankBicop()
{
    family_ = 5;
    family_name_ = "Frank";
    rotation_ = 0;
    association_direction_ = "both";
    parameters_ = VecXd::Zero(1);
    parameters_bounds_ = MatXd::Zero(1, 2);
    parameters_bounds_(0, 0) = -200.0;
    parameters_bounds_(0, 1) = 200.0;
}

FrankBicop::FrankBicop(const VecXd& parameters)
{
    FrankBicop();
    set_parameters(parameters);
}

FrankBicop::FrankBicop(const VecXd& parameters, const int& rotation)
{
    FrankBicop();
    set_parameters(parameters);
    set_rotation(rotation);
}

VecXd FrankBicop::generator(const VecXd& u)
{
    double theta = double(this->parameters_(0));
    auto f = [theta](const double v) {
        return (-1)*std::log((std::exp(-theta*v)-1)/(std::exp(-theta)-1));
    };
    return u.unaryExpr(f);
}
VecXd FrankBicop::generator_inv(const VecXd& u)
{
    double theta = double(this->parameters_(0));
    auto f = [theta](const double v) {
        //return (-1/theta)*std::log(1+std::exp(-theta*v)-std::exp(-v));
        return (-1/theta)*std::log(1+std::exp(-theta-v)-std::exp(-v));
    };
    return u.unaryExpr(f);
}

VecXd FrankBicop::generator_derivative(const VecXd& u)
{
    double theta = double(this->parameters_(0));
    auto f = [theta](const double v) {
        return theta/(1-std::exp(theta*v));
    };
    return u.unaryExpr(f);
}

VecXd FrankBicop::generator_derivative2(const VecXd& u)
{
    double theta = double(this->parameters_(0));
    auto f = [theta](const double v) {
        return std::pow(theta,2)/std::pow(std::exp(theta*v/2) - std::exp(-theta*v/2), 2);
    };
    return u.unaryExpr(f);
}

VecXd FrankBicop::tau_to_parameters(const double& tau)
{
    VecXd tau2 = VecXd::Constant(1, std::fabs(tau));
    auto f = [&](const VecXd &v) {
        return VecXd::Constant(1, std::fabs(parameters_to_tau(v)));
    };
    return invert_f(tau2, f, -100+1e-6, 100);
}

double FrankBicop::parameters_to_tau(const VecXd& parameters)
{
    double par = parameters(0);
    double tau = 1 - 4/par;
    double d = debyen(std::fabs(par), 1) / std::fabs(par);
    if (par < 0)
        d = d - par/2;
    tau = tau + (4/par) * d;
    return tau;
}

VecXd FrankBicop::get_start_parameters(const double tau)
{
    return tau_to_parameters(tau);
}

/*// PDF
VecXd FrankBicop::pdf_default(const MatXd& u)
{
    double theta = double(this->parameters_(0));
    MatXd t = u.unaryExpr([theta](const double v){ return std::exp(theta*v);});
    VecXd t1 = t.rowwise().prod();
    VecXd f = theta*(std::exp(theta)-1)*std::exp(theta)*t1;

    t1 = t1 - std::exp(theta)*(t.rowwise().sum() - VecXd::Ones(u.rows()));
    t1 = t1.array().square();

    f = f.cwiseQuotient(t1);
    return f;
}

// hfunction
VecXd FrankBicop::hfunc1_default(const MatXd& u)
{
    double theta = double(this->parameters_(0));
    MatXd t = u.unaryExpr([theta](const double v){ return std::exp(theta*v);});
    VecXd t1 = t.rowwise().prod();
    VecXd f = std::exp(theta)*(t.col(1) - VecXd::Ones(u.rows()));

    t1 = - t1 + std::exp(theta)*(t.rowwise().sum() - VecXd::Ones(u.rows()));
    f = f.cwiseQuotient(t1);
    return f;
}*/
