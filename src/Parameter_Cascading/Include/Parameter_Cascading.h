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
			 
			 Real angle_old_;
			 Real intensity_old_;
			 Real b1_old_;
			 Real b2_old_;
			 Real c_old_;

			 Real angle_;
			 Real intensity_;
			 Real b1_;
			 Real b2_;
			 Real c_;

			 VectorXr lambdas_;

	public: template<UInt ORDER, UInt mydim, UInt ndim>
			Parameter_Cascading(const PDE_Parameter_Functional<... Extension>& H,  const Real& angle_old, const Real& intensity_old,
								const Real& b1_old, const Real& b2_old, const Real& c_old,
								const MeshHandler<ORDER, mydim, ndim> & mesh)
			: H_(H), angle_old_(angle_old), intensity_old_(intensity_old), b1_old_(b1_old), b2_old_(b2_old), c_old_(c_old)
			{
				angle_ = angle_old_;
				intensity_ = intensity_old_;
				b1_ = b1_old_;
				b2_ = b2_old_;
				c_ = c_old_;
			
				VectorXd rhos;
				rhos = VectorXd::LinSpaced(0.01, 0, 1);
									
				Uint n = H.get_carrier() -> get_n_obs();
									
				Real area = 0.0;
				for(UInt i = 0u; i < mesh.num_elements(); ++i)
					area += mesh.elementMeasure(i);
							
				lambdas_ = rho.array() / (1 - rho.array()) * n / area;
			};
			
			void step_K();
			
			void step_b();
			
			void step_c();
			
			

};

#include "Parameter_Cascading_imp.h"

#endif
