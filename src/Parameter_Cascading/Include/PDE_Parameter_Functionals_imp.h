#ifndef __PDE_PARAMETER_FUNCTIONALS_IMP_H__
#define __PDE_PARAMETER_FUNCTIONALS_IMP_H__

#include <cmath> 
#include <limits>
#include <type_traits>
#include <utility>
#include "../../Regression/Include/Regression_Data.h"
#include "../../Lambda_Optimization/Include/Carrier.h"
#include "../../Lambda_Optimization/Include/Lambda_Optimizer.h"
#include "../../Lambda_Optimization/Include/Solution_Builders.h"

template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_K(const Real& angle, const Real& intensity) const
{
	// build the diffusion matrix from angle and intensity
	MatrixXr Q(2,2);
	Q << std::cos(angle), -std::sin(angle),
		 std::sin(angle), std::cos(angle);

	MatrixXr Sigma(2,2);
	Sigma << 1/std::sqrt(intensity), 0.,
 			 0., std::sqrt(intensity);

	MatrixXr K_matrix(2,2);
	K_matrix = Q * Sigma * Q.inverse();

	Rprintf("New K computed\n");

	// set the diffusion in RegressionData
	model.getRegressionData().getK().setDiffusion(K_matrix);

	Rprintf("New K set in RegressionData\n");
	Rprintf("Preapply\n");

	model.template preapply<ORDER, mydim, ndim>(mesh);

	return;
}
			

template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_b(const Real& b1, const Real& b2) const
{
	// build the advection vector from its components
	VectorXr b_vector(2,1);
	b_vector << b1, b2;
	
	// set the advection in RegressionData
	model.getRegressionData().getBeta().setAdvection(b_vector);

	model.template preapply<ORDER, mydim, ndim>(mesh);

	return;
}


template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_c(const Real& c) const
{	
	// set the advection in RegressionData
	//auto& regression_data = model.getRegressionData();

	//if constexpr (std::is_same<RegressionDataElliptic, decltype(regression_data) >::value) // constexpr KO in c++11
	//	regression_data.setC(c);
	
	//if constexpr (std::is_same<RegressionDataEllipticSpaceVarying, decltype(regression_data)>::value)
	//	regression_data.getC().setReaction(c);

	// PREAPPLY NEEDED

	return;
}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda) const
{
	// Check for proper values of angle and intensity
	// Notice that we keep angle in [0.0, EIGEN_PI] exploiting the periodicity of the matrix K
	if (angle < 0.0 || angle > EIGEN_PI || intensity <= 0.0)
		return std::numeric_limits<Real>::infinity();
	
	else
	{
		// set parameter in RegressionData
		set_K(angle, intensity);
		
		// solve the regression problem
		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
		
		GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> GS(carrier, true); // "true" only the first time would be better
		
		carrier.get_opt_data() -> set_current_lambdaS(lambda); // dovrebbe farlo già in carrier.apply

		GS.update_parameters(lambda);

		VectorXr z_hat = GS.get_z_hat();
		VectorXr zp = *(model.getRegressionData().getObservations());
		Real res = (zp - z_hat).squaredNorm();
		Rprintf("Result of eval_k: %f\n", res);

		return res;
    }

}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda) const
{
	// set parameter in RegressionData
	set_b(b1, b2);
	
	// solve the regression problem
	Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
		
	GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> GS(carrier, true);
		
	GS.update_parameters(lambda);
	VectorXr z_hat = GS.get_z_hat();
	VectorXr zp = *(model.getRegressionData().getObservations());
	return (zp - z_hat).squaredNorm();

}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_c(const Real& c, const lambda::type<1>& lambda) const
{
	// set parameter in RegressionData
	set_c(c);

	// solve the regression problem
	Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
		
	GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> GS(carrier, true);
		
	GS.update_parameters(lambda);
	VectorXr z_hat = GS.get_z_hat();
	VectorXr zp = *(model.getRegressionData().getObservations());
	return (zp - z_hat).squaredNorm();

}


template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda, const Real& h) const
{
	VectorXr res(2,1);

	// Check if angle and intensity remain in a proper range after finite difference schemes
	Real angle_lower, angle_upper, intensity_lower;
	Real h_angle_upper, h_angle_lower, h_intensity;
	
	if(angle - h < 0.0)
	{
		angle_lower = angle;
		h_angle_lower = h;
	}
	else
	{
		angle_lower = angle - h;
		h_angle_lower = 2. * h;
	}
	
	if(angle + h > EIGEN_PI)
	{
		angle_upper = angle;
		h_angle_upper = h;
	}
	else
	{
		angle_upper = angle + h;
		h_angle_upper = 2. * h;
	}
	
	if(intensity - h <= 0.0)
	{
		intensity_lower = intensity;
		h_intensity = h;
	}
	else
	{
		intensity_lower = intensity - h;
		h_intensity = 2. * h;
	}
	res << (eval_K(angle_upper, intensity, lambda) - eval_K(angle_lower, intensity, lambda)) / std::min(h_angle_upper, h_angle_lower),
		   (eval_K(angle, intensity + h, lambda) - eval_K(angle, intensity_lower, lambda)) / h_intensity;
	
	return res;
}


template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda, const Real& h) const
{
	VectorXr res(2,1);
	
	res << (eval_b(b1 + h, b2, lambda) - eval_b(b1 - h, b2, lambda)) / (2. * h),
		   (eval_b(b1, b2 + h, lambda) - eval_b(b1, b2 - h, lambda)) / (2. * h);
	
	return res;

}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_c(const Real& c, const lambda::type<1>& lambda, const Real& h) const
{
	return (eval_c(c + h, lambda) - eval_c(c - h, lambda)) / (2. * h);
}


#endif