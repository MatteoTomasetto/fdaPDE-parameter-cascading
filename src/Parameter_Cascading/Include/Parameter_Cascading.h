#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

template <typename InputCarrier>
class Parameter_Cascading {

	private: PDE_Parameter_Functional<InputCarrier> H;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K;
			 bool update_b;
			 bool update_c;
			 
			 // Boolean that is false only when we are stuck in a minimum during the optimization step;
			 // in particular it is false if:
			 // increment = || K - K-old || + || b - b_old|| || c - c_old|| < tol_parameter_cascading
			 bool goOn = true;
			 
			 // Diffusion parameters to optimize
			 Real angle;
			 Real intensity;
			 // Advection components to optimize
			 Real b1;
			 Real b2;
			 // Reaction coefficient to optimize
			 Real c;

			 Real increment = 0.0;

			 VectorXr lambdas; // lambdas used to search the optimal PDE parameters
			 
			 const Uint max_iter_parameter_cascading = 200;	// max number of iterations
			 const Real tol_parameter_cascading = 1e-3;		// tolerance 
			 
			 void step_K(); // find and update diffusion parameters via optimization algorithm
			 void step_b(); // find and update advection components via optimization algorithm
			 void step_c(); // find and update reaction parameter via optimization algorithm

	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			template<UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<InputCarrier>& H_,  bool update_K_, bool update_b_, update_c_, 
								const Real& angle_, const Real& intensity_, const Real& b1_, const Real& b2_, const Real& c_,
								const MeshHandler<ORDER, mydim, ndim> & mesh)
			: H(H_), update_K(update_K_), update_b(update_b_), update_c(update_c_),
			  angle(angle_), intensity(intensity_), b1(b1_), b2(b2_), c(c_)
			{
				// compute the lambdas for the parameter cascading algorithm from the rhos introduced in \cite{Bernardi}
				VectorXd rhos;
				rhos = VectorXd::LinSpaced(0.01, 0, 1);

				Uint n = H.get_solver().get_carrier().get_n_obs();

				Real area = 0.0;
				for(UInt i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;
			};
			
			// Function to apply the parameter cascading algorithm
			bool apply();

};

#include "Parameter_Cascading_imp.h"

#endif
