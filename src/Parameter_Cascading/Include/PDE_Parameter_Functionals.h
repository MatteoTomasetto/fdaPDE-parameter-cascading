#ifndef __PDE_PARAMETER_FUNCTIONALS_H__
#define __PDE_PARAMETER_FUNCTIONALS_H__

#include "../../FdaPDE.h"
#include "../../Lambda_Optimization/Include/Lambda_Optimizer.h"
#include "../../FE_Assemblers_Solvers/Include/Param_Functors.h"

/* *** PDE_Parameter_Functional ***
 *
 * Class PDE_Paramter_Functional compute the value of the functional that we need to optimize in order to select
 * the optimal values of the PDE_Paramters. It is the squared error done while approximating the observed data z_i with
 * z_hat_i through regression. 
 *
 * Notice that, since z_hat_i is computed using the PDE parameters, the functional depends implicitly on the PDE parameters.
 *
*/

// TODO For now we do not consider spacevarying case / GAM / temporal case (with lambdaS and lambdaT).

template <typename InputCarrier>
class PDE_Parameter_Functional
{
	private: GCV_Exact<InputCarrier, 1> & solver; // Lambda optimizer object with a carrier needed to solve the regression problem
			 MatrixXr K_matrix; // Diffusion matrix needed to build a Diffusion object from diffusion angle and intensity
			 VectorXr b_vector; // Advection vector needed to build Advection object from advection components

	public: // Constructor to set the pointer to the Carrier
			PDE_Parameter_Functional(GCV_Exact<InputCarrier, 1> & solver_) : GCV_Exact<InputCarrier, 1>(solver_);

			// Functions to build PDE parameters and set them in RegressionData
			void set_K(const Real& angle, const Real& intensity);
			void set_b(const Real& b1, const Real& b2);
			void set_c(const Real& c) const;

			// Functions to retrieve the value of the functional in the given input
			Real eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda));
			Real eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda);
			Real eval_c(const Real& c, const lambda::type<1>& lambda)) const;
			
			// Functions to retrieve the derivatives of the functional approximated via finite differences
			VectorXr eval_grad_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda),
								 const Real& h = 1e-3);
			VectorXr eval_grad_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda, const Real& h = 1e-3);
			Real	 eval_grad_c(const Real& c, const lambda::type<1>& lambda), const Real& h = 1e-3) const;
			
			// GETTERS
			inline GCV_Exact<InputCarrier, 1>& const get_solver(void) const { return solver; };

};

#include "PDE_Parameter_Functionals_imp.h"

#endif
