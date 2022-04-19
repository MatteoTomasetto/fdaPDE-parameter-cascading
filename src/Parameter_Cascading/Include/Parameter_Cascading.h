#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

template <typename ... Extensions>
class Parameter_Cascading {

	private: PDE_Parameter_Functional<... Extension> H_;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K_;
			 bool update_b_;
			 bool update_c_;
			 
			 // Boolean that is false only when we are stuck in a minimum; in particular it is false if
			 // increment = || K - K-old || + || b - b_old|| || c - c_old|| < tol_parameter_cascading
			 bool goOn_ = true;
			 
			 // Diffusion parameters to optimize
			 Real angle_;
			 Real intensity_;
			 // Advection components to optimize
			 Real b1_;
			 Real b2_;
			 // Reaction coefficient to optimize
			 Real c_;

			 Real increment_ = 0.0;

			 VectorXr lambdas_; // lambdas used to search the optimal PDE parameters
			 
			 const Uint max_iter_parameter_cascading_ = 300;	// max number of iterations
			 const Real tol_parameter_cascading_ = 1e-3;		// tolerance 
			 
			 void step_K(); // find and update diffusion parameters via optimization algorithm
			 void step_b(); // find and update advection components via optimization algorithm
			 void step_c(); // find and update reaction parameter via optimization algorithm

	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			template<UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<... Extension>& H,  bool update_K, bool update_b, update_c, 
								const Real& angle, const Real& intensity, const Real& b1, const Real& b2, const Real& c,
								const MeshHandler<ORDER, mydim, ndim> & mesh)
			: H_(H), update_K_(update_K), update_b_(update_b), update_c_(update_c),
			  angle_(angle), intensity_(intensity), b1_(b1), b2_(b2), c_(c)
			{
				VectorXd rhos;
				rhos = VectorXd::LinSpaced(0.01, 0, 1);
									
				Uint n = H_.get_carrier() -> get_n_obs();
									
				Real area = 0.0;
				for(UInt i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);
							
				lambdas_ = rhos.array() / (1 - rhos.array()) * n / area;
			};
			
			// Function to apply the parameter cascading algorithm
			bool apply();

};

#include "Parameter_Cascading_imp.h"

#endif
