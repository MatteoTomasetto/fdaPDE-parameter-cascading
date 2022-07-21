#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include "../../Lambda_Optimization/Include/Solution_Builders.h"

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
			 bool update_K_main_direction;
			 bool update_K_eigenval_ratio;
			 bool update_b;
			 bool update_c;

			 // Optimization algorithm to use: 0 for Gradient Descent, 1 for Genetic algorithm
			 UInt optimization_algorithm;
			 
			 // Diffusion parameters (angle/main_direction and intensity/eigenval_ratio)
			 VectorXr diffusion;
			 
			 // Advection parameter
			 VectorXr b;
			 
			 // Reaction coefficient
			 Real c;

			 VectorXr lambdas; // lambdas used to search the optimal PDE parameters
			 Real GCV;
			 Real lambda_opt; // Optimal lambda for GCV
			 
			 // Function to compute the optimal lambda through GCV
			 std::pair<Real, Real> compute_GCV(Carrier<RegressionDataElliptic>& carrier,
			 								   GCV_Exact<Carrier<RegressionDataElliptic>, 1>& solver,
			 								   Real lambda_init) const;
			 
			 VectorXr step(VectorXr init, const VectorXr& lower_bound, const VectorXr& upper_bound, const VectorXr& periods,
			 				const std::function<Real (VectorXr, Real)>& F, const std::function<VectorXr (VectorXr, Real)>& dF, 
			 				const std::function<void (VectorXr)>& set_param); 

	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			Parameter_Cascading(PDE_Parameter_Functional<ORDER, mydim, ndim>& H_)
			: H(H_) 
			{
				// Compute the lambdas for the parameter cascading algorithm from the rhos introduced in \cite{Bernardi}
				VectorXr rhos(13,1);
				rhos << 0.01,0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9;

				unsigned int n = H.getModel().getRegressionData().getNumberofObservations();

				Real area = 0.0;
				const MeshHandler<ORDER, mydim, ndim> & mesh = H.getMesh();
				for(unsigned int i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;

				// Initialize the parameters with the values in RegressionData and OptimizationData
				diffusion.resize(2);
				b.resize(2);
				diffusion(0) = H.getModel().getRegressionData().getK().getAngle();
				diffusion(1) = H.getModel().getRegressionData().getK().getIntensity();
				b(0) = H.getModel().getRegressionData().getB().get_b1_coeff();
				b(1) = H.getModel().getRegressionData().getB().get_b2_coeff();
				c = H.getModel().getRegressionData().getC().get_c_coeff();
				lambda_opt = H.getModel().getOptimizationData().get_initial_lambda_S();

				// Set which parameters to update
				update_K = false;
				update_K_main_direction = false;
				update_K_eigenval_ratio = false;
				update_b = false;
				update_c = false;
				UInt parameter_cascading_option = H.getModel().getRegressionData().get_parameter_cascading_option();

				if(parameter_cascading_option == 1){
					update_K = true;
				}
				
				if(parameter_cascading_option == 2){
					update_K_main_direction = true;
				}
				
				if(parameter_cascading_option == 3){
					update_K_eigenval_ratio = true;
				}

				// Other cases not implemented yet

				optimization_algorithm = H.getModel().getRegressionData().get_parameter_cascading_optimization_option();
			};
			
			// Function to apply the parameter cascading algorithm
			Output_Parameter_Cascading apply(void);
};

#include "Parameter_Cascading_imp.h"

#endif