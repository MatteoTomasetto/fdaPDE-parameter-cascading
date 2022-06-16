#ifndef __PARAM_FUNCTORS_H__
#define __PARAM_FUNCTORS_H__

#include "Pde_Expression_Templates.h"
#include <cmath>

// Forward declaration!
template <UInt ORDER, UInt mydim, UInt ndim>
class FiniteElement;

// Convenience enum for options
enum class PDEParameterOptions{Constant, SpaceVarying};

template<PDEParameterOptions OPTION>
class Diffusion{
public:

  Diffusion(Real* const K_ptr) :
    K_ptr_(K_ptr) {}

  Diffusion(SEXP RGlobalVector) :
    K_ptr_(REAL(RGlobalVector)) {}

  template<UInt ORDER, UInt mydim, UInt ndim>
  Real operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const;

  void setDiffusion(const MatrixXr & K) const
  { 
    Rprintf("Setting K in R: %f, %f, %f, %f", K(0,0), K(1,0), K(1,0), K(1,1));
    K_ptr_[0] = K(0,0);
    K_ptr_[1] = K(1,0);
    K_ptr_[2] = K(0,1);
    K_ptr_[3] = K(1,1);
  }

  Real getAngle(void) const
  { 
    Real intensity = getIntensity();

    if(std::abs(K_ptr_[1]) < 1e-16)
      return EIGEN_PI / 2;
    else
      return std::atan((K_ptr_[3] - std::sqrt(intensity)) / K_ptr_[1]);
  }

  Real getIntensity(void) const
  { 
    Real intensity = K_ptr_[3] + K_ptr_[0] + std::sqrt((K_ptr_[3] + K_ptr_[0]) * (K_ptr_[3] + K_ptr_[0]) + 4. * K_ptr_[1] * K_ptr_[1] - 4. * K_ptr_[3] * K_ptr_[0]);

    intensity /= 2.;

    return intensity*intensity;
  }


private:
  Real* const K_ptr_;
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Diffusion<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const {
  using EigenMap2Diff_matr = Eigen::Map<const Eigen::Matrix<Real,ndim,ndim> >;

  return fe_.stiff_impl(iq, i, j, EigenMap2Diff_matr(K_ptr_));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Diffusion<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const {
  using EigenMap2Diff_matr = Eigen::Map<const Eigen::Matrix<Real,ndim,ndim> >;

  const UInt index = fe_.getGlobalIndex(iq) * EigenMap2Diff_matr::SizeAtCompileTime;
  return fe_.stiff_impl(iq, i, j, EigenMap2Diff_matr(&K_ptr_[index]));
}

template<PDEParameterOptions OPTION>
class Advection{
public:

  Advection(Real* const b_ptr) :
    b_ptr_(b_ptr) {}

  Advection(SEXP RGlobalVector) :
    b_ptr_(REAL(RGlobalVector)) {}
    
  template<UInt ORDER, UInt mydim, UInt ndim>
  Real operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const;

  EOExpr<const Advection&> dot(const EOExpr<Grad>& grad) const {
    typedef EOExpr<const Advection&> ExprT;
    return ExprT(*this);
  }

  void setAdvection(const VectorXr & b) const
  {
    b_ptr_[0] = b(0);
    b_ptr_[1] = b(1);
  }

  Real get_b1_coeff(void) const
  { 
    return b_ptr_[0];
  }

  Real get_b2_coeff(void) const
  { 
    return b_ptr_[1];
  }

private:
  Real* const b_ptr_;
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Advection<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const {
  using EigenMap2Adv_vec = Eigen::Map<const Eigen::Matrix<Real,ndim,1> >;

  return fe_.grad_impl(iq, i, j, EigenMap2Adv_vec(b_ptr_));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Advection<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const {
  using EigenMap2Adv_vec = Eigen::Map<const Eigen::Matrix<Real,ndim,1> >;

  const UInt index = fe_.getGlobalIndex(iq) * EigenMap2Adv_vec::SizeAtCompileTime;
  return fe_.grad_impl(iq, i, j, EigenMap2Adv_vec(&b_ptr_[index]));
}


template<PDEParameterOptions OPTION>
class Reaction{
public:

	Reaction(Real* const  c_ptr) :
		c_ptr_(c_ptr) {}

	Reaction(SEXP RGlobalVector) :
    c_ptr_(REAL(RGlobalVector)) {}

    
  template<UInt ORDER, UInt mydim, UInt ndim>
  Real operator() (const FiniteElement<ORDER, mydim, ndim>& fe_, UInt iq, UInt i, UInt j) const;

  EOExpr<const Reaction&> operator* (const EOExpr<Mass>&  mass) const {
      typedef EOExpr<const Reaction&> ExprT;
      return ExprT(*this);
  }

  void setReaction(const Real & c) const { *c_ptr_ = c;}

  Real get_c_coeff(void) const
  { 
    return *c_ptr_;
  }

private:
  Real* const c_ptr_;
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Reaction<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER, mydim, ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  return c_ptr_[0]*fe_.mass_impl(iq, i, j);
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Reaction<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER, mydim, ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  const UInt index = fe_.getGlobalIndex(iq);
  return c_ptr_[index]*fe_.mass_impl(iq, i, j);
}


class ForcingTerm{
public:

  ForcingTerm() :
    u_ptr_(nullptr) {}
	ForcingTerm(const Real* const u_ptr) :
		u_ptr_(u_ptr) {}

	ForcingTerm(SEXP RGlobalVector) :
    u_ptr_(REAL(RGlobalVector)) {}

  template<UInt ORDER, UInt mydim, UInt ndim>
  Real integrate (const FiniteElement<ORDER, mydim, ndim>& fe_, UInt i) const {
    const UInt index = fe_.getGlobalIndex(0);
    return fe_.forcing_integrate(i, &u_ptr_[index]);
  }

private:
  const Real* const u_ptr_;
};


#endif /* PARAM_FUNCTORS_H_ */
