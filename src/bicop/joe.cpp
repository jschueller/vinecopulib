// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://tvatter.github.io/vinecopulib/.

#include "bicop/joe.hpp"
#include <boost/math/special_functions/digamma.hpp>

namespace vinecopulib
{
    JoeBicop::JoeBicop()
    {
        family_ = BicopFamily::joe;
        rotation_ = 0;
        parameters_ = Eigen::VectorXd(1);
        parameters_lower_bounds_ = Eigen::VectorXd(1);
        parameters_upper_bounds_ = Eigen::VectorXd(1);
        parameters_ << 1;
        parameters_lower_bounds_ << 1;
        parameters_upper_bounds_ << 200;
    }

    double JoeBicop::generator(const double& u)
    {
        return (-1)*std::log(1-std::pow(1-u, this->parameters_(0)));
    }

    double JoeBicop::generator_inv(const double& u)
    {
        return 1-std::pow(1-std::exp(-u),1/this->parameters_(0));
    }

    double JoeBicop::generator_derivative(const double& u)
    {
        double theta = double(this->parameters_(0));
        return (-theta)*std::pow(1-u, theta-1)/(1-std::pow(1-u, theta));
    }

    double JoeBicop::generator_derivative2(const double& u)
    {
        double theta = double(this->parameters_(0));
        return theta*(theta-1+std::pow(1-u, theta))*std::pow(1-u, theta-2)/std::pow(-1+std::pow(1-u, theta),2);
    }

    // inverse h-function
    Eigen::VectorXd JoeBicop::hinv1(
        const Eigen::Matrix<double, Eigen::Dynamic, 2>& u
    )
    {
        double theta = double(this->parameters_(0));
        double u1, u2;
        Eigen::VectorXd hinv = Eigen::VectorXd::Zero(u.rows());
        for (int j = 0; j < u.rows(); ++j) {
            u1 = u(j, 1);
            u2 = u(j, 0);
            hinv(j) = qcondjoe(&u1, &u2, &theta);
        }

        return hinv;
    }

    // link between Kendall's tau and the par_bicop parameter
    Eigen::MatrixXd JoeBicop::tau_to_parameters(const double& tau)
    {
        Eigen::VectorXd tau2 = Eigen::VectorXd::Constant(1, std::fabs(tau));
        auto f = [&](const Eigen::VectorXd &v) {
            return Eigen::VectorXd::Constant(1, std::fabs(parameters_to_tau(v)));
        };
        return invert_f(tau2, f, 1+1e-6, 100);
    }

    double JoeBicop::parameters_to_tau(const Eigen::VectorXd& parameters)
    {
        double par = parameters(0);
        double tau = 2 / par + 1;
        tau = boost::math::digamma(2.0) - boost::math::digamma(tau);
        tau = 1 + 2 * tau / (2 - par);
        return flip_tau(tau);
    }

    Eigen::VectorXd JoeBicop::get_start_parameters(const double tau)
    {
        return tau_to_parameters(tau);
    }
}

// This is copy&paste from the VineCopula package
double qcondjoe(double* q, double* u, double* de)
{
    double t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t13,t15,t16,t19,t23,t28,t31;
    double c21,pdf;
    int iter;
    double diff,v,de1,dtem,de1inv,tem;

    t1 = 1.0-*u;
    t2 = pow(t1,1.0*(*de));
    t7 = 1./(*de);
    t10 = t2*(*de);
    t11 = 1./t1;
    t19 = (*de)*(*de);
    de1=*de-1;  // may need better modification for large delta
    dtem=-de1/(1.+de1); de1inv=-1./de1;

    // v = 0.5 * (q+u); // starting guess

    // Use a better starting point based on reflected B4 copula
    // A good starting point is crucial when delta is large because
    //    C_{2|1} will be steep
    // C_{R,2|1}(v|u)=1-C_{2|1}(1-v|1-u),
    // C_{R,2|1}^{-1}(q|u)=1-C_{2|1}^{-1}(1-q|1-u)
    tem=pow(1.-*q,dtem)-1.;
    tem=tem*pow(1.-*u,-de1)+1.;
    v=pow(tem,de1inv); v=1.-v;
    diff=1; iter=0;
    while(fabs(diff)>1.e-6 && iter<20)
    { t3 = 1.-v;
        t4 = pow(t3,*de);
        t5 = t2*t4;
        t6 = t2+t4-t5;
        t8 = pow(t6,t7);
        t9 = t7*t8;
        t13 = t11*t4;
        t15 = -t10*t11+t10*t13;
        t16 = 1./t6;
        t23 = 1./t3;
        t28 = t6*t6;
        t31 = (-t4*(*de)*t23+t5*(*de)*t23)/t28*t15;
        c21 = -t9*t15*t16;
        pdf = -t8/t19*t31+t8*(*de)*t2*t13*t23*t16+t9*t31;
        iter++;
        if(std::isnan(pdf) || std::isnan(c21) ) { diff/=-2.; }  // added for de>=30
        else diff=(c21-*q)/pdf;
        v-=diff;
        int iter2 = 0;
        while((v<=0 || v>=1 || fabs(diff)>0.25) & (iter2 <20)) {++iter2; diff/=2.; v+=diff; }
    }
    return(v);
}
