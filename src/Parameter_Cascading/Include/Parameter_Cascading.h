#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functional.h"
#include "../../Lambda_Optimization/Include/Solution_Builders.h"

/* *** Parameter_Cascading ***
 *
 * Class Parameter_Cascading performs the Parameter Cascading algorithm that aims to estimate the PDE_parameters (diffusion matrix K,
 * advection vector b and reaction coefficient c) minimizing the mean squared error of the approximation of z_i with
 * z_hat through regression. 
 *
*/

template <UInt ORDER, UInt mydim, UInt ndim, typename InputHandler>
class Parameter_Cascading
{
	private: // Functional to optimize in the Parameter Cascading algorithm (Mean Squared Error)
			 PDE_Parameter_Functional<ORDER, mydim, ndim, InputHandler> & H;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K = false;
			 bool update_anisotropy_intensity = false;
			 bool update_K_direction = false;
			 bool update_K_eigenval_ratio = false;
			 bool update_b = false;
			 bool update_b_direction = false;
			 bool update_b_intensity = false;
			 bool update_c = false;

			 MatrixXr K;
			 
			 // Diffusion parameters
			 // with 2D stationary cases the diffusion parameters are:
       		 //       1) the diffusion angle, i.e. the main eigenvector direction of K
    		 //       2) the diffusion intensity, i.e. the ratio between the second and the first eigenvalue of K
			 // with 3D stationary cases the diffusion parameters are:
			 //       1) the angle wrt z-axis
    		 //       2) the angle wrt y-axis
    		 //       3) the ratio between the first (biggest) and the third (smallest) eigenvalue of K
    		 //       4) the ratio between the second and the third (smallest) eigenvalue of K
			 VectorXr diffusion;

			 // Advection parameter
			 // 2D or 3D stationary case: polar coordinates (angles and module) of the advection vector
			 VectorXr advection;
			 
			 // Advection parameters
			 // 2D or 3D stationary case: advection vector
			 VectorXr b;
			 
			 // Reaction coefficient
			 // 2D or 3D stationary case: reaction coefficient
			 Real c;

			 // Anisotropy intensity
			 // 2D or 3D stationary case: coefficient that pre-multiply the diffusion matrix K
			 Real aniso_intensity;

			 VectorXr lambdas; // lambdas used to search the optimal PDE parameters
			 Real GCV;
			 Real lambda_opt; // Optimal lambda for GCV
			 
			 // Boolean to select which optimization procedure for GCV to use
			 bool stochastic_GCV = false;

			 // Functions to select and compute the optimal lambda and the GCV
			 std::pair<Real, Real> select_and_compute_GCV(void) const;
			 template <typename EvaluationType>
			 std::pair<Real, Real> compute_GCV(EvaluationType& solver) const;
			 			 
			 // Main function of the algorithm; inputs are the following:
			 // init -> starting point for optimization algorithm;
			 // Optimization algorithm to use: 0 for L-BFGS-B, 1 for BFGS, 2 for Conjugate Gradient, 3 for Nelder-Mead, 
			 // 4 for Gradient Descent, 5 for Genetic Algorithm;
			 // lower_bound -> vector of the lower bounds for each parameter to optimize;
			 // upper_bound -> vector of the upper bounds for each parameter to optimize;
			 // periods -> vector with the periods of each parameter to optimize (if a parameter is not periodic then period = 0.0);
			 // F -> function to optimize;
			 // dF -> gradient of F or approximated gradient via finite differences;
			 // set_param -> function to set the proper parameter in RegressionData;
			 // constraint -> boolean for contrained optimization;
			 VectorXr step(VectorXr init, const UInt& opt_algo, const VectorXr& lower_bound, const VectorXr& upper_bound, const VectorXr& periods,
			 				const std::function<Real (VectorXr, Real)>& F, const std::function<VectorXr (VectorXr, Real)>& dF, 
			 				const std::function<void (VectorXr)>& set_param, bool constraint = false);

			 // Alternative version of step() where lower bounds are -inf, upper bounds are +inf and periods are 0.0 
			 VectorXr step(VectorXr init, const UInt& opt_algo, const std::function<Real (VectorXr, Real)>& F, const std::function<VectorXr (VectorXr, Real)>& dF, 
			 				const std::function<void (VectorXr)>& set_param); 

	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			Parameter_Cascading(PDE_Parameter_Functional<ORDER, mydim, ndim, InputHandler>& H_)
			: H(H_) 
			{

				// Compute the lambdas for the parameter cascading algorithm from the rhos grid introduced in \cite{Bernardi}
				VectorXr rhos(13,1);
				rhos << 0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99;

				unsigned int n = H.getModel().getRegressionData().getNumberofObservations();

				Real area = 0.0;
				const MeshHandler<ORDER, mydim, ndim> & mesh = H.getMesh();
				for(unsigned int i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;

				// Initialize the parameters with the values in RegressionData and OptimizationData
				K = H.getModel().getRegressionData().getK().template getDiffusionMatrix<ndim>();
				diffusion = H.getModel().getRegressionData().getK().template getDiffusionParam<ndim>();
				b = H.getModel().getRegressionData().getB().template getAdvectionVector<ndim>();
				advection = H.getModel().getRegressionData().getB().template getAdvectionParam<ndim>();
				c = H.getModel().getRegressionData().getC().getReactionParam();
				aniso_intensity = (ndim == 2) ? std::sqrt(K.determinant()) : std::cbrt(K.determinant());
				lambda_opt = H.getModel().getOptimizationData().get_initial_lambda_S();

				// Set which parameters to update
				UInt parameter_cascading_diffusion_option = H.getModel().getRegressionData().get_parameter_cascading_diffusion();
				UInt parameter_cascading_advection_option = H.getModel().getRegressionData().get_parameter_cascading_advection();

				if(parameter_cascading_diffusion_option == 1){
					update_K = true;
				}
				
				if(parameter_cascading_diffusion_option == 2){
					update_K_direction = true;
				}
				
				if(parameter_cascading_diffusion_option == 3){
					update_K_eigenval_ratio = true;
				}

				if(parameter_cascading_diffusion_option == 4){
					update_anisotropy_intensity = true;
				}

				if(parameter_cascading_advection_option == 1){
					update_b = true;
				}

				if(parameter_cascading_advection_option == 2){
					update_b_direction = true;
				}

				if(parameter_cascading_advection_option == 3){
					update_b_intensity = true;
				}

				if(H.getModel().getRegressionData().get_parameter_cascading_reaction() == 1){
					update_c = true;
				}

				// GCV_Stochastic will be used if the user requires it
				OptimizationData& optr = H.getModel().getOptimizationData();
				if(optr.get_loss_function() == "GCV" && (optr.get_DOF_evaluation() == "stochastic" || optr.get_DOF_evaluation() == "not_required"))
				{
					stochastic_GCV = true;
				}
			};
			
			// Function to apply the parameter cascading algorithm
			Output_Parameter_Cascading apply(void);
};

#include "Parameter_Cascading_imp.h"

#endif