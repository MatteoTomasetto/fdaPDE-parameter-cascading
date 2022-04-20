#ifndef __PDE_PARAMETER_FUNCTIONALS_IMP_H__
#define __PDE_PARAMETER_FUNCTIONALS_IMP_H__

#include <cmath> 
#include <limits>

template <typename InputCarrier>
void PDE_Parameter_Functional<InputCarrier>::set_K(const Real& angle, const Real& intensity)
{
	MatrixXr Q;
	Q << std::cos(angle), -std::sin(angle),
		 std::sin(angle), std::cos(angle);

	MatrixXr Sigma;
	Sigma << 1/std::sqrt(intensity), 0.,
 			 0., std::sqrt(intensity);

	K_matrix = Q * Sigma * Q.inverse();

	const Diffusion<PDEParameterOption::Constant> K(K_matrix.data());

	solver.get_carrier().get_model() -> getRegressionData().setK(K);
				
	return;
}
			

template <typename InputCarrier>
void PDE_Parameter_Functional<InputCarrier>::set_b(const Real& b1, const Real& b2)
{
	b_vector = (VectorXr << b1, b2).finished()

	const Advection<PDEParameterOption::Constant> b(b_vector.data());

	solver.get_carrier().get_model() -> getRegressionData().setBeta(b);

	return;
}


template <typename InputCarrier>
void PDE_Parameter_Functional<InputCarrier>::set_c(const Real& c) const
{
	solver.get_carrier.get_model() -> getRegressionData().setC(c);
	
	return;
}


template <typename InputCarrier>
Real PDE_Parameter_Functional<InputCarrier>::eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda)
{
	// Check for proper values of angle and intensity
	// Notice that we keep angle in [0.0, EIGEN_PI] exploiting the periodicity of the matrix K
	if (angle < 0.0 || angle > EIGEN_PI || intensity <= 0.0)
		return std::numeric_limits<Real>::infinity();
	
	else
	{
		set_K(angle, intensity);
		
		solver.update_parameters(lambda)

		VectorXr z_hat = solver.get_z_hat();

		return (solver.get_carrier().get_zp()) - z_hat).squaredNorm();
    }
}


template <typename InputCarrier>
Real PDE_Parameter_Functional<InputCarrier>::eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda)
{
	set_b(b1, b2);
	
	solver.update_parameters(lambda);

	VectorXr z_hat = solver.get_z_hat();

	return (solver.get_carrier().get_zp()) - z_hat).squaredNorm();
    }
}


template <typename InputCarrier>
Real PDE_Parameter_Functional<InputCarrier>::eval_c(const Real& c, const lambda::type<1>& lambda)) const
{
	set_c(c);

	solver.update_parameters(lambda);

	VectorXr z_hat = solver.get_z_hat();

	return (solver.get_carrier().get_zp()) - z_hat).squaredNorm();
    }
}


template <typename InputCarrier>
VextorXr PDE_Parameter_Functional<InputCarrier>::eval_grad_K(const Real& angle, const Real& intensity,
															 const lambda::type<1>& lambda), const Real& h)
{
	VectorXr res;

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
	res << (eval_K(angle_upper, intensity, lambda) - eval_K(angle_lower, intensity, lambda)) / std::min(h_angle_upper, 																											h_angle_lower),
		   (eval_K(angle, intensity + h, lambda) - eval_K(angle, intensity_lower, lambda)) / h_intensity;
	
	return res;
}


template <typename InputCarrier>
VectorXr PDE_Parameter_Functional<InputCarrier>::eval_grad_b(const Real& b1, const Real& b2, 
														     const lambda::type<1>& lambda, const Real& h)
{
	VectorXr res;
	
	res << (eval_b(b1 + h, b2, lambda) - eval_b(b1 - h, b2, lambda)) / (2. * h),
		   (eval_b(b1, b2 + h, lambda) - eval_b(b1, b2 - h, lambda)) / (2. * h);
	
	return res;

}


template <typename InputCarrier>
Real PDE_Parameter_Functional<InpuCarrier>::eval_grad_c(const Real& c, const lambda::type<1>& lambda), const Real& h)
{
	return (eval_c(c + h, lambda) - eval_c(c - h, lambda)) / (2. * h);
}


#endif
