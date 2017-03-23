// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://tvatter.github.io/vinecopulib/.

#pragma once

#include "abstract.hpp"

namespace vinecopulib {
    class Bicop
    {
    public:
        // Constructors
        Bicop();
        Bicop(BicopFamily family, int rotation = 0);
        Bicop(BicopFamily family, int rotation, Eigen::VectorXd parameters);
        Bicop(
                Eigen::Matrix<double, Eigen::Dynamic, 2> data,
                std::vector<BicopFamily> family_set = bicop_families::all,
                std::string method = "mle",
                std::string selection_criterion = "bic",
                bool preselect_families = true
        );

        // Getters and setters
        BicopFamily get_family() const;
        std::string get_family_name() const;
        int get_rotation() const;
        std::string get_association_direction() const;
        Eigen::MatrixXd get_parameters() const;
        Eigen::MatrixXd get_parameters_lower_bounds() const;
        Eigen::MatrixXd get_parameters_upper_bounds() const;
        void set_rotation(const int& rotation);
        void set_parameters(const Eigen::MatrixXd& parameters);

        // Stats methods
        Eigen::VectorXd pdf(const Eigen::Matrix<double,Eigen::Dynamic,2>& u);
        Eigen::VectorXd hfunc1(const Eigen::Matrix<double,Eigen::Dynamic,2>& u);
        Eigen::VectorXd hfunc2(const Eigen::Matrix<double,Eigen::Dynamic,2>& u);
        Eigen::VectorXd hinv1(const Eigen::Matrix<double,Eigen::Dynamic,2>& u);
        Eigen::VectorXd hinv2(const Eigen::Matrix<double,Eigen::Dynamic,2>& u);
        Eigen::Matrix<double,Eigen::Dynamic,2> simulate(const int& n);

        // Methods modifying the family/rotation/parameters
        void fit(
                const Eigen::Matrix<double, Eigen::Dynamic, 2> &data,
                std::string method
        );
        void select(
                Eigen::Matrix<double, Eigen::Dynamic, 2> data,
                std::vector<BicopFamily> family_set = bicop_families::all,
                std::string method = "mle",
                std::string selection_criterion = "bic",
                bool preselect_families = true
        );
        void flip();

        // Fit statistics
        double loglik(const Eigen::Matrix<double, Eigen::Dynamic, 2>& u);
        double aic(const Eigen::Matrix<double, Eigen::Dynamic, 2>& u);
        double bic(const Eigen::Matrix<double, Eigen::Dynamic, 2>& u);

        // Misc
        double calculate_npars();
        double parameters_to_tau(const Eigen::VectorXd& parameters);
        Eigen::MatrixXd tau_to_parameters(const double& tau);

    private:
        BicopPtr get_bicop();
        BicopPtr bicop_;
    };
}
