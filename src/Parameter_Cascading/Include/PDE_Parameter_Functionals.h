#ifndef __PDE_PARAMETER_FUNCTIONALS_H__
#define __PDE_PARAMETER_FUNCTIONALS_H__

#include "../../FdaPDE.h"
#include "../../FE_Assemblers_Solvers/Include/Param_Functors.h"
#include "../../Regression/Include/Mixed_FE_Regression.h"

/* *** PDE_Parameter_Functional ***
 *
 * Class PDE_Paramter_Functional compute the value of the functional that we need to optimize in order to select
 * the optimal values of the PDE_Paramters. It is the squared error done while approximating the observed data z_i with
 * z_hat_i through regression. 
 *
 * Notice that, since z_hat_i is computed using the PDE parameters, the functional depends implicitly on the PDE parameters.
 *
*/

template <UInt ORDER, UInt mydim, UInt ndim>
class PDE_Parameter_Functional
{
	private: // MixedFERegression object with to solve the regression problem
			 MixedFERegression<RegressionDataElliptic> & model;

			 // Mesh needed to recompute R1 matrix every time the PDE parameters change
			 const MeshHandler<ORDER, mydim, ndim> & mesh;
			 
	public: // Constructor to set the reference to the solver
			PDE_Parameter_Functional(MixedFERegression<RegressionDataElliptic> & model_,
									 const MeshHandler<ORDER, mydim, ndim> & mesh_)
			: model(model_), mesh(mesh_) {};

			// Functions to build PDE parameters, set them in RegressionData and recompute R1 matrix 
			void set_K(const Real& angle, const Real& intensity) const;
			void set_b(const Real& b1, const Real& b2) const;
			void set_c(const Real& c) const;

			// Functions to retrieve the value of the functional in the given input
			Real eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda) const;
			Real eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda) const;
			Real eval_c(const Real& c, const lambda::type<1>& lambda) const;
			
			// Functions to retrieve the derivatives of the functional approximated via finite differences
			// (these could be useful for optimization methods)
			Eigen::Vector2d eval_grad_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			Eigen::Vector2d eval_grad_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			Real	 eval_grad_c(const Real& c, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			
			// GETTERS
			inline MixedFERegression<RegressionDataElliptic> & getModel(void) const { return model; };
			inline const MeshHandler<ORDER, mydim, ndim> & getMesh(void) const { return mesh; };
};

#include "PDE_Parameter_Functionals_imp.h"

#endif