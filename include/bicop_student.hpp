/*
* The MIT License (MIT)
*
* Copyright © 2017 Thibault Vatter and Thomas Nagler
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef VINECOPULIB_BICOP_STUDENT_HPP
#define VINECOPULIB_BICOP_STUDENT_HPP

#include "bicop_elliptical.hpp"

class StudentBicop : public EllipticalBicop {

public:
    // constructor
    StudentBicop();
    StudentBicop(const VecXd& parameters);
    StudentBicop(const VecXd& parameters, const int& rotation);

    // PDF
    VecXd pdf_default(const MatXd& u);

    // hfunction
    VecXd hfunc1_default(const MatXd& u);

    // inverse hfunction
    VecXd hinv1_default(const MatXd& u);

private:
    VecXd get_start_parameters(const double tau);
};


#endif
