#ifndef __PARAM_FUNCTORS_H__
#define __PARAM_FUNCTORS_H__

#include "Pde_Expression_Templates.h"
#include "../../FE_Assemblers_Solvers/Include/Finite_Element.h"
#include "../../Mesh/Include/Mesh.h"
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

  template<UInt ORDER, UInt mydim, UInt ndim>
  void set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh);

  void setDiffusion(const MatrixXr & K) const
  { 
    if((!(K.cols() == 2 && K.rows() == 2)) && (!(K.cols() == 3 && K.rows() == 3)))
    {
      Rf_error("Wrong diffusion matrix dimension in setDiffusion");
      abort();
    }
    else
    {
      UInt Kdim = K.cols();
      UInt counter;
      
      // Fill the matrix in Param Functors with the input
      for(UInt k = 0; k < dim; ++k)
      {
        counter = 0;

        for(UInt i = 0; i < Kdim; ++i)
        {
          for(UInt j = 0; j < Kdim; ++j)
          {
            K_ptr_[i + j + counter + k*Kdim*Kdim] = K(j,i); // Col-wise ordering
          }
        
          counter += Kdim - 1;
        }
      }
    }
  }

  void setDiffusion(const VectorXr& DiffParam) const
  {
    // Build the diffusion matrix from the diffusion parameters
    // if ndim == 2 then the parameters are:
    //       1) the diffusion angle, i.e. the main eigenvector direction of K
    //       2) the diffusion intensity, i.e. the ratio between the second and the first eigenvalue of K
    if((DiffParam.size() != 2) && (DiffParam.size() != 4))
    {
      Rf_error("Wrong diffusion parameters vector dimension in setDiffusion");
      abort();
    }
    if(DiffParam.size() == 2) // case with ndim = 2
    {
      // Use the parametrization of K through the diffusion parameters
      MatrixXr Q(2,2);
      Q << std::cos(DiffParam(0)), -std::sin(DiffParam(0)),
           std::sin(DiffParam(0)), std::cos(DiffParam(0));

      MatrixXr Sigma(2,2);
      Sigma << 1/std::sqrt(DiffParam(1)), 0.,
               0., std::sqrt(DiffParam(1));

      MatrixXr K = Q * Sigma * Q.transpose();

      setDiffusion(K);
    }
    // if ndim == 3 then the parameters are:
    //       1) the rotation angle alpha wrt z-axis
    //       2) the rotation angle beta wrt y-axis
    //       3) the ratio between the first (biggest) and the third (smallest) eigenvalue of K
    //       4) the ratio between the second and the third (smallest) eigenvalue of K
    else if(DiffParam.size() == 4) // case with ndim = 3
    {
      // Use the parametrization of K through the diffusion parameters
      MatrixXr Qz(3,3);
      Qz << std::cos(DiffParam(0)), -std::sin(DiffParam(0)), 0.0,
            std::sin(DiffParam(0)), std::cos(DiffParam(0)), 0.0,
            0.0, 0.0, 1.0;

      MatrixXr Qy(3,3);
      Qy << std::cos(DiffParam(1)), 0.0, std::sin(DiffParam(1)),
            0.0, 1.0, 0.0,
            -std::sin(DiffParam(1)), 0.0, std::cos(DiffParam(1));

      MatrixXr Sigma(3,3);
      Sigma << std::cbrt(DiffParam(2) * DiffParam(2) / DiffParam(3)), 0.0, 0.0,
               0.0, std::cbrt(DiffParam(3) * DiffParam(3) / DiffParam(2)), 0.0,
               0.0, 0.0, std::cbrt( 1.0 / (DiffParam(2) * DiffParam(3)));

      MatrixXr Q = Qz * Qy;

      MatrixXr K = Q * Sigma * Q.transpose();

      setDiffusion(K);
    }

    return;
  }
    
  // Compute diffusion parameters from K
  template<UInt ndim>
  MatrixXr getDiffusionMatrix(void) const
  { 
    MatrixXr K(ndim, ndim);

    UInt counter = 0;

    // Save diffusion matrix coefficient in matrix K and return K
    for(UInt i = 0; i < ndim; ++i)
    {
      for(UInt j = 0; j < ndim; ++j)
      {
        K(j,i) = K_ptr_[i + j + counter];
      }
        
      counter += ndim - 1;
    }

    return K;
  }
   
  // Compute diffusion parameters from K
  template<UInt ndim>
  VectorXr getDiffusionParam(void) const
  { 
    UInt dim = (ndim == 2) ? 2 : 4; // Diffusion paramter dimension

    VectorXr res(dim); // Vector to store the result (the diffusion parameters)

    if(ndim == 2) // if ndim == 2 then the parameters are diffusion angle and diffusion intensity
    {
      // Exploit the inverse formulas to retrieve the diffusion parameters from K
      Real intensity = K_ptr_[3] + K_ptr_[0] + std::sqrt((K_ptr_[3] + K_ptr_[0]) * (K_ptr_[3] + K_ptr_[0]) + 4. * K_ptr_[1] * K_ptr_[1] - 4. * K_ptr_[3] * K_ptr_[0]);

      intensity /= 2.;

      res(1) = intensity*intensity;

      if(std::abs(K_ptr_[1]) < 1e-16)
        res(0) = EIGEN_PI / 2;
      else
      {
        Real angle = std::atan((K_ptr_[3] - std::sqrt(intensity)) / K_ptr_[1]);
      
        res(0) = (angle < 0.0) ? angle + EIGEN_PI : angle;
      }
    }

    if(ndim == 3) // if ndim == 3 then the parameters are 3 diffusion angles and 2 diffusion intensities
    {
      // Exploit the inverse formulas to retrieve the diffusion parameters from K
      MatrixXr K = getDiffusionMatrix<ndim>();
      Eigen::EigenSolver<MatrixXr> K_eigen(K);
     
      VectorXr K_eigenvalues = K_eigen.eigenvalues().real();
      
      Eigen::Index maxpos, minpos, midpos;
      K_eigenvalues.maxCoeff(&maxpos);
      K_eigenvalues.minCoeff(&minpos);

      if(minpos != 0 && maxpos != 0)
        midpos = 0;
      else if(minpos != 1 && maxpos != 1)
        midpos = 1;
      else
        midpos = 2;

      res(2) = K_eigenvalues(maxpos)/K_eigenvalues(minpos); // first eigenvalue ratio
      res(3) = K_eigenvalues(midpos)/K_eigenvalues(minpos); // second eigenvalue ratio

      VectorXr v1 = K_eigen.eigenvectors().col(maxpos).real();
      VectorXr v2 = K_eigen.eigenvectors().col(midpos).real();
      VectorXr v3 = K_eigen.eigenvectors().col(minpos).real();

      if(v1(2) >= 0.0)
        v1 *= (-1.0); // Change sign to have beta in [0, EIGEN_PI]

      Real beta1 = std::asin(-v1(2));
      Real beta2 = EIGEN_PI - beta1;

      Real alpha1 = std::acos(v1(0)/std::cos(beta1));
      Real alpha2 = std::acos(v1(0)/std::cos(beta2));

      if(std::abs(v1(1) - std::cos(beta1)*std::sin(alpha1)) < 1e-6)
      {
        res(0) = alpha1; // rotation angle wrt z axis
        res(1) = beta1; // rotation angle wrt y axis
      }
      else
      {
        res(0) = alpha2;  // rotation angle wrt z axis
        res(1) = beta2;  // rotation angle wrt y axis
      }
    }

    return res;
  }

private:
  Real* const K_ptr_;

  UInt dim = 1; // Number of pointed matrices
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Diffusion<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  using EigenMap2Diff_matr = Eigen::Map<const Eigen::Matrix<Real,ndim,ndim> >;

  return fe_.stiff_impl(iq, i, j, EigenMap2Diff_matr(K_ptr_));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Diffusion<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  using EigenMap2Diff_matr = Eigen::Map<const Eigen::Matrix<Real,ndim,ndim> >;

  const UInt index = fe_.getGlobalIndex(iq) * EigenMap2Diff_matr::SizeAtCompileTime;
  return fe_.stiff_impl(iq, i, j, EigenMap2Diff_matr(&K_ptr_[index]));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Diffusion<PDEParameterOptions::Constant>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = 1;
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Diffusion<PDEParameterOptions::SpaceVarying>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = FiniteElement<ORDER, mydim, ndim>::Integrator::NNODES * mesh.num_elements();
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

  EOExpr<const Advection&> dot(const EOExpr<Grad>& grad) const
  {
    typedef EOExpr<const Advection&> ExprT;
    return ExprT(*this);
  }

  template<UInt ORDER, UInt mydim, UInt ndim>
  void set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh);

  void setAdvection(const VectorXr & b) const
  {
    if(b.size() != 2 && b.size() != 3)
    {
      Rf_error("Wrong advection vector dimension in setAdvection");
      abort();
    }
    else
    {
      // Set advection vector equal to the input
      for(UInt j = 0; j < dim; ++j)
        for(UInt i = 0; i < b.size(); ++i)
          b_ptr_[i + j*b.size()] = b(i);
      
    }

    return;
  }

  void setAdvectionParam(const VectorXr& AdvectionParam) const 
  {
    VectorXr b(AdvectionParam.size());

    if(AdvectionParam.size() == 2)
    {
      b(0) = AdvectionParam(1) * std::cos(AdvectionParam(0));
      b(1) = AdvectionParam(1) * std::sin(AdvectionParam(0));
    }

    if(AdvectionParam.size() == 3)
    {
      b(0) = AdvectionParam(2) * std::cos(AdvectionParam(0)) * std::sin(AdvectionParam(1));
      b(1) = AdvectionParam(2) * std::sin(AdvectionParam(0)) * std::sin(AdvectionParam(1));
      b(2) = AdvectionParam(2) * std::cos(AdvectionParam(1));
    }

    setAdvection(b);

    return;
  }

  // Get advection parameters, that are simply the coefficients inside the advection vector
  template<UInt ndim>
  VectorXr getAdvectionVector(void) const 
  {
    VectorXr res(ndim);

    for(UInt i = 0; i < ndim; ++i)
        res(i) = b_ptr_[i];

    return res;
  }

  // Get advection parameters, that are simply the coefficients inside the advection vector
  template<UInt ndim>
  VectorXr getAdvectionParam(void) const 
  {
    VectorXr res(ndim);

    if(ndim == 2)
    {
      res(1) = std::sqrt(b_ptr_[0]*b_ptr_[0] + b_ptr_[1]*b_ptr_[1]);

      res(0) = std::atan2(b_ptr_[1],b_ptr_[0]);

      if(res(0) < 0.0)
        res(0) += 2.0 * EIGEN_PI;
    }

    if(ndim == 3)
    {
      res(2) = std::sqrt(b_ptr_[0]*b_ptr_[0] + b_ptr_[1]*b_ptr_[1] + b_ptr_[2]*b_ptr_[2]);

      res(0) = std::atan2(b_ptr_[1],b_ptr_[0]);

      if(res(0) < 0.0)
        res(0) += 2.0 * EIGEN_PI;

      if(std::abs(res(2)) < 1e-6)
        res(1) = 0;
      else
      {
        res(1) = std::acos(b_ptr_[2] / res(2));
      }
      
    }

    return res;
  }

private:
  Real* const b_ptr_;

  UInt dim = 1; // Number of pointed vectors
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Advection<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  using EigenMap2Adv_vec = Eigen::Map<const Eigen::Matrix<Real,ndim,1> >;

  return fe_.grad_impl(iq, i, j, EigenMap2Adv_vec(b_ptr_));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Advection<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  using EigenMap2Adv_vec = Eigen::Map<const Eigen::Matrix<Real,ndim,1> >;

  const UInt index = fe_.getGlobalIndex(iq) * EigenMap2Adv_vec::SizeAtCompileTime;
  return fe_.grad_impl(iq, i, j, EigenMap2Adv_vec(&b_ptr_[index]));
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Advection<PDEParameterOptions::Constant>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = 1;
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Advection<PDEParameterOptions::SpaceVarying>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = FiniteElement<ORDER, mydim, ndim>::Integrator::NNODES * mesh.num_elements();
}


template<PDEParameterOptions OPTION>
class Reaction{
public:

	Reaction(Real* const c_ptr) :
		c_ptr_(c_ptr) {}

	Reaction(SEXP RGlobalVector) :
    c_ptr_(REAL(RGlobalVector)) {}

  template<UInt ORDER, UInt mydim, UInt ndim>
  Real operator() (const FiniteElement<ORDER, mydim, ndim>& fe_, UInt iq, UInt i, UInt j) const;

  EOExpr<const Reaction&> operator* (const EOExpr<Mass>&  mass) const
  {
      typedef EOExpr<const Reaction&> ExprT;
      return ExprT(*this);
  }

  template<UInt ORDER, UInt mydim, UInt ndim>
  void set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh);

  void setReaction(const Real & c) const
  {
    for(UInt i = 0; i < dim; ++i)
      c_ptr_[i] = c;
  }

  // Get reaction parameter, that is direclty the reaction coefficient c
  Real getReactionParam(void) const
  { 
    return c_ptr_[0];
  }

private:
  Real* const c_ptr_;

  UInt dim = 1; // Number of pointed numbers
};

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Reaction<PDEParameterOptions::Constant>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  return c_ptr_[0]*fe_.mass_impl(iq, i, j);
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
Real Reaction<PDEParameterOptions::SpaceVarying>::operator() (const FiniteElement<ORDER,mydim,ndim>& fe_, UInt iq, UInt i, UInt j) const
{
  const UInt index = fe_.getGlobalIndex(iq);
  return c_ptr_[index]*fe_.mass_impl(iq, i, j);
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Reaction<PDEParameterOptions::Constant>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = 1;
}

template<>
template<UInt ORDER, UInt mydim, UInt ndim>
void Reaction<PDEParameterOptions::SpaceVarying>::set_dim(const MeshHandler<ORDER, mydim, ndim> & mesh)
{
  dim = FiniteElement<ORDER, mydim, ndim>::Integrator::NNODES * mesh.num_elements();
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
