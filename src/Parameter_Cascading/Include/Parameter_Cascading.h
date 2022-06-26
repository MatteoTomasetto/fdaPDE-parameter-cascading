#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

/* *** Parameter_Cascading ***
 *
 * Class Parameter_Cascading performs the Parameter Cascading algorithm that aims to estimate the PDE_parameters (diffusion matrix K,
 * advection vector b and reaction coefficient c) minimizing the mean squared error of the approximation of z_i with
 * z_hat through regression. 
 *
*/

template <UInt ORDER, UInt mydim, UInt ndim>
class Parameter_Cascading
{
	private: // Functional to optimize in the Parameter Cascading algorithm (mean squared error)
			 PDE_Parameter_Functional<ORDER, mydim, ndim> & H;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K;
			 bool update_b;
			 bool update_c;
			 
			 // Diffusion parameters
			 Real angle;
			 Real intensity;
			 // Advection components
			 Real b1;
			 Real b2;
			 // Reaction coefficient
			 Real c;

			 VectorXr lambdas; // lambdas used to search the optimal PDE parameters
			 Real lambda_opt; // Optimal lambda for GCV
			 
			 // Function to compute the optimal lambda through GCV
			 std::pair<Real, Real> compute_GCV(Carrier<RegressionDataElliptic>& carrier,
			 								   GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>& solver,
			 								   Real lambda_init) const;
			 
			 void step_K(void); // Find and update diffusion parameters via optimization algorithm
			 void step_b(void); // Find and update advection components via optimization algorithm
			 void step_c(void); // Find and update reaction parameter via optimization algorithm
			 
	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			Parameter_Cascading(PDE_Parameter_Functional<ORDER, mydim, ndim>& H_)
			: H(H_) 
			{
				// Compute the lambdas for the parameter cascading algorithm from the rhos introduced in \cite{Bernardi}
				VectorXr rhos;
				rhos = VectorXr::LinSpaced(3, 0.01, 0.99); // TO FIX

				unsigned int n = H.getModel().getRegressionData().getNumberofObservations();

				Real area = 0.0;
				const MeshHandler<ORDER, mydim, ndim> & mesh = H.getMesh();
				for(unsigned int i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;

				// Initialize the parameters with the values in RegressionData and OptimizationData
				angle = H.getModel().getRegressionData().getK().getAngle();
				intensity = H.getModel().getRegressionData().getK().getIntensity();
				b1 = H.getModel().getRegressionData().getB().get_b1_coeff();
				b2 = H.getModel().getRegressionData().getB().get_b2_coeff();
				c = H.getModel().getRegressionData().getC().get_c_coeff();
				Real lambda_opt = H.getModel().getOptimizationData().get_initial_lambda_S();

				// Set which parameters to update
				update_K = false;
				update_b = false;
				update_c = false;
				UInt parameter_cascading_option = H.getModel().getRegressionData().get_parameter_cascading_option();

				if(parameter_cascading_option == 1)
					update_K = true;
				// Other cases not implemented yet
			};
			
			// Function to apply the parameter cascading algorithm
			Real apply(void);

};

#include "Parameter_Cascading_imp.h"

#endif