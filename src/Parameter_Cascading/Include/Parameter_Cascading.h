#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

template <UInt ORDER, UInt mydim, UInt ndim>
class Parameter_Cascading
{
	private: // Functional to optimize in the parameter cascading algorithm
			 PDE_Parameter_Functional<ORDER, mydim, ndim> & H;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K;
			 bool update_b;
			 bool update_c;
			 
			 // Diffusion parameters to optimize
			 Real angle;
			 Real intensity;
			 // Advection components to optimize
			 Real b1;
			 Real b2;
			 // Reaction coefficient to optimize
			 Real c;

			 VectorXr lambdas; // lambdas used to search the optimal PDE parameters
			 
			 // Function to compute the optimal lambda through GCV
			 std::pair<Real, Real> compute_optimal_lambda(Carrier<RegressionDataElliptic>& carrier, GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>& GS, Real lambda_init) const;
			 
			 void step_K(void); // Find and update diffusion parameters via optimization algorithm
			 void step_b(void); // Find and update advection components via optimization algorithm
			 void step_c(void); // Find and update reaction parameter via optimization algorithm
			 
	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			Parameter_Cascading(PDE_Parameter_Functional<ORDER, mydim, ndim>& H_)
			: H(H_) 
			{
				// compute the lambdas for the parameter cascading algorithm from the rhos introduced in \cite{Bernardi}
				VectorXr rhos;
				rhos = VectorXr::LinSpaced(3, 0.01, 0.99); // set length vector

				unsigned int n = H.getModel().getRegressionData().getNumberofObservations();

				Real area = 0.0;
				const MeshHandler<ORDER, mydim, ndim> & mesh = H.getMesh();

				for(unsigned int i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				Rprintf("area computed from mesh = %e\n", area);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;

				// initialize the parameters with the values in RegressionData
				angle = H.getModel().getRegressionData().getK().getAngle();
				intensity = H.getModel().getRegressionData().getK().getIntensity();
				b1 = H.getModel().getRegressionData().getB().get_b1_coeff();
				b2 = H.getModel().getRegressionData().getB().get_b2_coeff();
				c = H.getModel().getRegressionData().getC().get_c_coeff();

				// set which parameters to update
				update_K = false;
				update_b = false;
				update_c = false;
				UInt parameter_cascading_option = H.getModel().getRegressionData().get_parameter_cascading_option();
				if(parameter_cascading_option == 1)
					update_K = true;
			};
			
			// Function to apply the parameter cascading algorithm
			void apply(void);

};

#include "Parameter_Cascading_imp.h"

#endif