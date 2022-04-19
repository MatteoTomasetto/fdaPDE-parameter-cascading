#ifndef __PARAMETER_CASCADING_H__
#define __PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

template <typename ... Extensions>
class Parameter_Cascading {

	private: PDE_Parameter_Functional<... Extension> H_;

			 bool update_K = true;
			 bool update_b = true;
			 bool update_c = true;
			 
			 Real angle_;
			 Real intensity_;
			 Real b1_;
			 Real b2_;
			 Real c_;

			 VectorXr lambdas_; // lambdas used to search the optimal PDE parameters
			 
			 Uint max_iter_parameter_cascading; // max number of iterations
			 Real tol_parameter_cascading;		// tolerance 
			 
			void step_K();

			void step_b();

			void step_c();

	public: template<UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<... Extension>& H,  const Real& angle, const Real& intensity,
								const Real& b1, const Real& b2, const Real& c,
								const MeshHandler<ORDER, mydim, ndim> & mesh)
			: H_(H), angle_(angle), intensity_(intensity), b1_(b1), b2_(b2), c_(c)
			{
				VectorXd rhos;
				rhos = VectorXd::LinSpaced(0.01, 0, 1);
									
				Uint n = H_.get_carrier() -> get_n_obs();
									
				Real area = 0.0;
				for(UInt i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);
							
				lambdas_ = rhos.array() / (1 - rhos.array()) * n / area;
			};
			
			void apply();

};

#include "Parameter_Cascading_imp.h"

#endif
