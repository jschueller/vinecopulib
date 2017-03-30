// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://tvatter.github.io/vinecopulib/.

#include <vinecopulib/misc/tools_integration.hpp>
#include <boost/numeric/odeint.hpp>

namespace tools_integration {

    double integrate_zero_to_one(std::function<double(double)> f)
    {
        boost::numeric::odeint::runge_kutta_dopri5<double> stepper;
        double lb = 1e-12;
        double ub = 1.0 - lb;
        double x = 0.0;
        auto ifunc = [f](const double /* x */, double &dxdt, const double t) {
            dxdt = f(t);
        };
        integrate_adaptive(boost::numeric::odeint::make_controlled(lb,lb,stepper),
                           ifunc, x, lb, ub, lb);
        return x;
    }
}
