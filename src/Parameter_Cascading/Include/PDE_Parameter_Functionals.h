#ifndef __PDE_PARAMETER_FUNCTIONALS_H__
#define __PDE_PARAMETER_FUNCTIONALS_H__

#include "../../FdaPDE.h"
#include "../../Lambda_Optimization/Include/Carrier.h"
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

template <typename ... Extensions>
class PDE_Parameter_Functional
{
	private: Carrier<RegressionDataElliptic, ... Extensions> * carrier_; // Carrier needed to solve the regression problem
			 MatrixXr K_matrix_; // Diffusion matrix needed to build a Diffusion object from diffusion angle and intensity
			 VectorXr b_vector_; // Advection vector needed to build Advection object from advection components
			 

	public: // Constructor to set the pointer to the Carrier
			PDE_Parameter_Functional(Carrier<RegressionDataElliptic, ... Extensions> * carrier) : carrier_(carrier);
			
			// Function to build the diffusion matrix K starting from diffusion angle α (alpha) and diffusion intensity γ (gamma)
			const Diffusion<PDEParameterOptions::Constant> build_K(const Real& angle, const Real& intensity);
			
			// Function to build the advection vector b starting from its 2 components
			const Advection<PDEParameterOptions::Constant> build_b(const Real& b1, const Real& b2);

			// Functions to retrieve the value of the functional in the given input
			Real eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda)) const;
			Real eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda) const;
			Real eval_c(const Real& c, const lambda::type<1>& lambda)) const;
			
			// Functions to retrieve the derivatives of the functional approximated via finite differences
			VectorXr eval_grad_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda),
								 const Real& h = 1e-3) const;
			VectorXr eval_grad_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda, const Real& h = 1e-3) const;
			Real	 eval_grad_c(const Real& c, const lambda::type<1>& lambda), const Real& h = 1e-3) const;
			
			// GETTERS
			inline Carrier<RegressionDataElliptic, ... Extensions>* get_carrier() const { return this -> carrier_;};

			//SETTERS
			inline void set_K_matrix(const Real& angle, const Real& intensity)
			{
				MatrixXr Q;
				Q << std::cos(angle), -std::sin(angle),
		 			 std::sin(angle), std::cos(angle);
		 		
		 		MatrixXr Sigma;
		 		Sigma << 1/std::sqrt(intensity), 0.,
			 			 0., std::sqrt(intensity);

				this -> K_matrix_ = Q * Sigma * Q.inverse();
				
				return;
			};
			
			inline void set_b_vector(const Real& b1, const Real& b2)
			{
				this -> b_vector_ = (VectorXr << b1, b2).finished()
				
				return;
			};
};

#include "PDE_Parameter_Functionals_imp.h"

#endif
