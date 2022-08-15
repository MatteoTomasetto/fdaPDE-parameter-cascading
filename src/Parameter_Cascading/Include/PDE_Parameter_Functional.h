#ifndef __PDE_PARAMETER_FUNCTIONAL_H__
#define __PDE_PARAMETER_FUNCTIONAL_H__

#include "../../FdaPDE.h"
#include "../../FE_Assemblers_Solvers/Include/Param_Functors.h"
#include "../../Regression/Include/Mixed_FE_Regression.h"
#include "../../Global_Utilities/Include/Lambda.h"

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

			template <typename DiffType> // VectorXr (diffusion parameters) and MatrixXr (diffusion matrix) are allowed
			typename std::enable_if<std::is_same<DiffType, VectorXr>::value || std::is_same<DiffType, MatrixXr>::value, void>::type
			set_K(const DiffType& DiffParam) const;
			
			void set_b(const VectorXr& AdvParam) const;
			void set_c(const Real& c) const;

			// Functions to retrieve the value of the functional in the given input
			template <typename DiffType> // VectorXr (diffusion parameters) and MatrixXr (diffusion matrix) are allowed
			typename std::enable_if<std::is_same<DiffType, VectorXr>::value || std::is_same<DiffType, MatrixXr>::value, Real>::type
			eval_K(const DiffType& DiffParam, const lambda::type<1>& lambda) const;
			Real eval_b(const VectorXr& AdvParam, const lambda::type<1>& lambda) const;
			Real eval_c(const Real& c, const lambda::type<1>& lambda) const;
			
			// Functions to retrieve the derivatives of the functional approximated via centered finite differences
			// (these could be useful for optimization methods)
			VectorXr eval_grad_K(const VectorXr& DiffParam, const VectorXr& LowerBound, const VectorXr& UpperBound, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			VectorXr eval_grad_b(const VectorXr& AdvParam, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			VectorXr eval_grad_c(const Real& c, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			VectorXr eval_grad_aniso_intensity(const MatrixXr& K, const Real& aniso_intensity, const lambda::type<1>& lambda, const Real& h = 1e-3) const;

			// GETTERS
			inline MixedFERegression<RegressionDataElliptic> & getModel(void) const { return model; };
			inline const MeshHandler<ORDER, mydim, ndim> & getMesh(void) const { return mesh; };
};

#include "PDE_Parameter_Functional_imp.h"

#endif