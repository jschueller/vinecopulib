// Copyright © 2017 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://tvatter.github.io/vinecopulib/.

#pragma once

#include "test_tools.hpp"
#include "gtest/gtest.h"
#include "rscript.hpp"

class VinecopTest : public ::testing::Test {
public:
    VinecopTest();
    MatXd u;
    VecXd f;
    MatXd sim;
    MatXi model_matrix;
    MatXi vc_matrix;
};
