//  Copyright (C) 2018 Yi Pan <ypan1988@gmail.com>
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available at
//  https://www.R-project.org/Licenses/

#ifndef __ROPTIM_H__
#define __ROPTIM_H__

#include <cassert>
#include <cmath>
#include <cstddef>

#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../FdaPDE.h"
#include <R_ext/Applic.h> // optimization algorithm by stats::optim()

namespace roptim {

struct OptStruct {
  bool has_grad_ = false;
  bool has_hess_ = false;
  VectorXr ndeps_;       // tolerances for numerical derivatives
  double fnscale_ = 1.0;  // scaling for objective
  VectorXr parscale_;    // scaling for parameters
  int usebounds_ = 0;
  VectorXr lower_, upper_;
  bool sann_use_custom_function_ = false;
};

class Functor {
 public:
  Functor() {
  }

  virtual ~Functor() {}

  virtual double operator()(const VectorXr &par) = 0;
  virtual void Gradient(const VectorXr &par, VectorXr &grad) {
    ApproximateGradient(par, grad);
  }
  virtual void Hessian(const VectorXr &par, MatrixXr &hess) {
    ApproximateHessian(par, hess);
  }

  // Returns forward-difference approximation of Gradient
  void ApproximateGradient(const VectorXr &par, VectorXr &grad);

  // Returns forward-difference approximation of Hessian
  void ApproximateHessian(const VectorXr &par, MatrixXr &hess);

  OptStruct os;
};

inline void Functor::ApproximateGradient(const VectorXr &par,
                                         VectorXr &grad) {
  if (os.parscale_.size()==0) os.parscale_ = VectorXr::Constant(par.size(), 1.0);
  if (os.ndeps_.size()==0)
    os.ndeps_ = VectorXr::Constant(par.size(), 1e-3);

  grad = VectorXr::Zero(par.size());
  VectorXr x = par.array() * os.parscale_.array();

  if (os.usebounds_ == 0) {
    for (std::size_t i = 0; i != par.size(); ++i) {
      double eps = os.ndeps_(i);

      x(i) = (par(i) + eps) * os.parscale_(i);
      double val1 = operator()(x) / os.fnscale_;

      x(i) = (par(i) - eps) * os.parscale_(i);
      double val2 = operator()(x) / os.fnscale_;

      grad(i) = (val1 - val2) / (2 * eps);

      x(i) = par(i) * os.parscale_(i);
    }
  } else {  // use bounds
    for (std::size_t i = 0; i != par.size(); ++i) {
      double epsused = os.ndeps_(i);
      double eps = os.ndeps_(i);

      double tmp = par(i) + eps;
      if (tmp > os.upper_(i)) {
        tmp = os.upper_(i);
        epsused = tmp - par(i);
      }

      x(i) = tmp * os.parscale_(i);
      double val1 = operator()(x) / os.fnscale_;

      tmp = par(i) - eps;
      if (tmp < os.lower_(i)) {
        tmp = os.lower_(i);
        eps = par(i) - tmp;
      }

      x(i) = tmp * os.parscale_(i);
      double val2 = operator()(x) / os.fnscale_;

      grad(i) = (val1 - val2) / (epsused + eps);

      x(i) = par(i) * os.parscale_(i);
    }
  }
}

inline void Functor::ApproximateHessian(const VectorXr &par, MatrixXr &hess) {
  if (os.parscale_.size()==0) os.parscale_ = VectorXr::Constant(par.size(), 1.0);
  if (os.ndeps_.size()==0)
    os.ndeps_ = VectorXr::Constant(par.size(), 1e-3);

  hess = MatrixXr::Zero(par.size(), par.size());
  VectorXr dpar = par.array() / os.parscale_.array();
  VectorXr df1 = VectorXr::Zero(par.size());
  VectorXr df2 = VectorXr::Zero(par.size());

  for (std::size_t i = 0; i != par.size(); ++i) {
    double eps = os.ndeps_(i) / os.parscale_(i);
    dpar(i) += eps;
    Gradient(dpar, df1);
    dpar(i) -= 2 * eps;
    Gradient(dpar, df2);
    for (std::size_t j = 0; j != par.size(); ++j)
      hess(i, j) = os.fnscale_ * (df1(j) - df2(j)) /
                   (2 * eps * os.parscale_(i) * os.parscale_(j));
    dpar(i) = dpar(i) + eps;
  }

  // now symmetrize
  for (std::size_t i = 0; i != par.size(); ++i) {
    for (std::size_t j = 0; j != par.size(); ++j) {
      double tmp = 0.5 * (hess(i, j) + hess(j, i));

      hess(i, j) = tmp;
      hess(j, i) = tmp;
    }
  }
}

inline double fminfn(int n, double *x, void *ex) {
  OptStruct os(static_cast<Functor *>(ex)->os);

  VectorXr par(n);
  for(int i=0; i<n; ++i)
    par(i) = x[i];
  par = par.array() * os.parscale_.array();
  return static_cast<Functor *>(ex)->operator()(par) / os.fnscale_;
}

inline void fmingr(int n, double *x, double *gr, void *ex) {
  OptStruct os(static_cast<Functor *>(ex)->os);

  VectorXr par(n);
  for(int i=0; i<n; ++i)
    par(i) = x[i];

  VectorXr grad;
  par = par.array() * os.parscale_.array();
  static_cast<Functor *>(ex)->Gradient(par, grad);
  for (int i = 0; i != n; ++i)
    gr[i] = grad(i) * (os.parscale_(i) / os.fnscale_);
}

// Wrapper to stats::optim functionality of R
template <typename Derived>
class Roptim {
 public:
  std::string method_;
  VectorXr lower_, upper_;
  bool hessian_flag_ = false;
  MatrixXr hessian_;

  VectorXr lower() const { return lower_; }
  VectorXr upper() const { return upper_; }

  VectorXr par() const { return par_; }
  double value() const { return val_; }
  int fncount() const { return fncount_; }
  int grcount() const { return grcount_; }
  int convergence() const { return fail_; }
  std::string message() const { return message_; }
  MatrixXr hessian() const { return hessian_; }

 private:
  VectorXr par_;
  double val_ = 0.0;
  int fncount_ = 0;
  int grcount_ = 0;
  int fail_ = 0;
  std::string message_ = "NULL";

 public:
  struct RoptimControl {
    std::size_t trace = 0;
    double fnscale = 1.0;
    VectorXr parscale;
    VectorXr ndeps;
    std::size_t maxit = 100;
    double abstol = std::numeric_limits<Real>::min();
    double reltol = std::sqrt(2.220446e-16);
    double alpha = 1.0;
    double beta = 0.5;
    double gamma = 2.0;
    int REPORT = 10;
    bool warn_1d_NelderMead = true;
    int type = 1;
    int lmm = 5;
    double factr = 1e7;
    double pgtol = 0.0;
    double temp = 10.0;
    int tmax = 10;
  } control;

  Roptim(const std::string method = "Nelder-Mead") : method_(method) {
    if (method_ != "Nelder-Mead" && method_ != "BFGS" && method_ != "CG" &&
        method_ != "L-BFGS-B")
      Rf_error("Roptim::Roptim(): unknown 'method'");

    // Sets default value for maxit & REPORT (which depend on method)
    if (method_ == "Nelder-Mead") {
      control.maxit = 500;
    }
  }

  void set_method(const std::string &method) {
    if (method != "Nelder-Mead" && method != "BFGS" && method != "CG" &&
        method != "L-BFGS-B")
      Rf_error("Roptim::set_method(): unknown 'method'");
    else
      method_ = method;

    // Sets default value for maxit & REPORT (which depend on method)
    if (method_ == "Nelder-Mead") {
      control.maxit = 500;
      control.REPORT = 10;
    } else {
      control.maxit = 100;
      control.REPORT = 10;
    }
  }

  void set_lower(const VectorXr &lower) {
    if (method_ != "L-BFGS-B")
      Rprintf("Roptim::set_lower(): bounds can only be used with method L-BFGS-B");
    method_ = "L-BFGS-B";
    lower_ = lower;
  }

  void set_upper(const VectorXr &upper) {
    if (method_ != "L-BFGS-B")
      Rprintf("Roptim::set_upper(): bounds can only be used with method L-BFGS-B");
    method_ = "L-BFGS-B";
    upper_ = upper;
  }

  void set_hessian(bool flag) { hessian_flag_ = flag; }

  void minimize(Derived &func, VectorXr &par);
};

template <typename Derived>
inline void Roptim<Derived>::minimize(Derived &func, VectorXr &par) {
  // PART 1: optim()

  // Checks if lower and upper bounds is used
  if ((lower_.size() == 0 || !upper_.size() == 0) && method_ != "L-BFGS-B") {
    Rprintf("bounds can only be used with method L-BFGS-B");
    method_ = "L-BFGS-B";
  }

  // Sets the parameter size
  std::size_t npar = par.size();

  // Sets default value for parscale & ndeps (which depend on npar)
  if (control.parscale.size() == 0)
    control.parscale = VectorXr::Constant(npar, 1.0);
  if (control.ndeps.size() == 0)
    control.ndeps = VectorXr::Constant(npar, 1e-3);

  // Note that "method L-BFGS-B uses 'factr' (and 'pgtol') instead of 'reltol'
  // and 'abstol'". There is no simple way to detect whether users set new
  // values for 'reltol' and 'abstol'.

  // Gives warning of 1-dim optimization by Nelder-Mead
  if (npar == 1 && method_ == "Nelder-Mead" && control.warn_1d_NelderMead)
    Rprintf("one-dimensional optimization by Nelder-Mead is unreliable");

  // Sets default value for lower_
  if (method_ == "L-BFGS-B" && lower_.size()==0) {
    lower_ = VectorXr::Constant(npar, std::numeric_limits<Real>::min());
  }
  // Sets default value for upper_
  if (method_ == "L-BFGS-B" && upper_.size()==0) {
    upper_ = VectorXr::Constant(npar, std::numeric_limits<Real>::max());
  }

  // PART 2: C_optim()

  func.os.usebounds_ = 0;
  func.os.fnscale_ = control.fnscale;
  func.os.parscale_ = control.parscale;

  if (control.ndeps.size() != npar)
    Rf_error("'ndeps' is of the wrong length");
  else
    func.os.ndeps_ = control.ndeps;

  VectorXr dpar = VectorXr::Zero(npar);
  VectorXr opar = VectorXr::Zero(npar);

  dpar = par.array() / control.parscale.array();

  if (method_ == "Nelder-Mead") {
    nmmin(npar, dpar.data(), opar.data(), &val_, fminfn, &fail_,
          control.abstol, control.reltol, &func, control.alpha, control.beta,
          control.gamma, control.trace, &fncount_, control.maxit);

    par = opar.array() * control.parscale.array();
    grcount_ = 0;

  } else if (method_ == "BFGS") {
    Eigen::VectorXi mask = Eigen::VectorXi::Constant(npar, 1);
    vmmin(npar, dpar.data(), &val_, fminfn, fmingr, control.maxit,
          control.trace, mask.data(), control.abstol, control.reltol,
          control.REPORT, &func, &fncount_, &grcount_, &fail_);

    par = dpar.array() * control.parscale.array();
  } else if (method_ == "CG") {
    cgmin(npar, dpar.data(), opar.data(), &val_, fminfn, fmingr, &fail_,
          control.abstol, control.reltol, &func, control.type, control.trace,
          &fncount_, &grcount_, control.maxit);

    par = opar.array() * control.parscale.array();
  } else if (method_ == "L-BFGS-B") {
    VectorXr lower(npar);
    VectorXr upper(npar);
    Eigen::VectorXi nbd = Eigen::VectorXi::Zero(npar);
    char msg[60];

    for (std::size_t i = 0; i != npar; ++i) {
      lower(i) = lower_(i) / func.os.parscale_(i);
      upper(i) = upper_(i) / func.os.parscale_(i);
      if (!std::isfinite(lower(i))) {
        if (!std::isfinite(upper(i)))
          nbd(i) = 0;
        else
          nbd(i) = 3;
      } else {
        if (!std::isfinite(upper(i)))
          nbd(i) = 1;
        else
          nbd(i) = 2;
      }
    }

    func.os.usebounds_ = 1;
    func.os.lower_ = lower;
    func.os.upper_ = upper;

    lbfgsb(npar, control.lmm, dpar.data(), lower.data(), upper.data(),
           nbd.data(), &val_, fminfn, fmingr, &fail_, &func, control.factr,
           control.pgtol, &fncount_, &grcount_, control.maxit, msg,
           control.trace, control.REPORT);

    par = dpar.array() * control.parscale.array();
    message_ = msg;
  } else
    Rf_error("Roptim::minimize(): unknown 'method'");

  par_ = par;
  val_ *= func.os.fnscale_;

  if (hessian_flag_) func.ApproximateHessian(par_, hessian_);
}

}  // namespace roptim

#endif  // ROPTIM_H_
