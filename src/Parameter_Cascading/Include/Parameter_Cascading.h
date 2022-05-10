#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

template <typename InputCarrier>
class Parameter_Cascading
{
	private: // Functional to optimize in the parameter cascading algorithm
			 PDE_Parameter_Functional<InputCarrier> & H;

			 // Booleans to keep track of the wanted parameters to optimize
			 bool update_K;
			 bool update_b;
			 bool update_c;
			 
			 // Boolean that is false only when we are stuck in a minimum during the optimization step;
			 // in particular it is false if:
			 // increment = || K - K-old || + || b - b_old|| + || c - c_old|| < tol_parameter_cascading
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
			 
			 const unsigned int max_iter_parameter_cascading;	// max number of iterations
			 const Real tol_parameter_cascading;				// tolerance 
			 
			 // function to compute the optimal lambda through GCV
			 Real compute_optimal_lambda(void) const;
			 
			 void step_K(void); // find and update diffusion parameters via optimization algorithm
			 void step_b(void); // find and update advection components via optimization algorithm
			 void step_c(void); // find and update reaction parameter via optimization algorithm
			 
	public: // Constructor that computes the vector of lambdas from the vector of rhos presented in \cite{Bernardi}
			template<class DType, class CType, UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<InputCarrier>& H_,
								bool update_K_, bool update_b_, bool update_c_, 
								const Real& angle_, const Real& intensity_, const Real& b1_, const Real& b2_, const Real& c_,
								const MeshHandler<ORDER, mydim, ndim> & mesh, const unsigned int& max_iter_parameter_cascading_, 
								const Real & tol_parameter_cascading_)
			: H(H_), update_K(update_K_), update_b(update_b_), update_c(update_c_),
			  angle(angle_), intensity(intensity_), b1(b1_), b2(b2_), c(c_), 
			  max_iter_parameter_cascading(max_iter_parameter_cascading_), 
			  tol_parameter_cascading(tol_parameter_cascading_) 
			{
				// compute the lambdas for the parameter cascading algorithm from the rhos introduced in \cite{Bernardi}
				VectorXr rhos;
				rhos = VectorXr::LinSpaced(0.01, 0, 1);

				unsigned int n = H.get_solver().get_carrier().get_n_obs();

				Real area = 0.0;
				for(unsigned int i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);

				lambdas = rhos.array() / (1 - rhos.array()) * n / area;
			};
			
			// Constructor that set max_iter and tolerance with default values 
			template<class DType, class CType, UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<InputCarrier>& H_,
								bool update_K_, bool update_b_, bool update_c_, 
								const Real& angle_, const Real& intensity_, const Real& b1_, const Real& b2_, const Real& c_,
								const MeshHandler<ORDER, mydim, ndim> & mesh)
			: Parameter_Cascading(H_, update_K_, update_b_, update_c_, angle_, intensity_, b1_, b2_, c_, mesh, 200u, 1e-3) {};


			// Function to apply the parameter cascading algorithm
			bool apply(void);

};

#include "Parameter_Cascading_imp.h"

#endif