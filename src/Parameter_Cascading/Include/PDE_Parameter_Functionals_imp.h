#ifndef __PDE_PARAMETER_FUNCTIONALS_IMP_H__
#define __PDE_PARAMETER_FUNCTIONALS_IMP_H__

#include <cmath> 
#include <limits>

template <typename ...Extension>
const Diffusion<PDEParameterOption::Constant> PDE_Parameter_Functional<...Extension>::build_K(const Real& angle,
																							  const Real& intensity)
{

	set_K_matrix(angle, intesity);

	const Diffusion<PDEParameterOption::Constant> K((this -> K_matrix_).data());

	return K;
}

template <typename ...Extension>
const Advection<PDEParameterOption::Constant> PDE_Parameter_Functional<...Extension>::build_b(const Real& b1, const Real& b2)
{
	set_b_vector(b1, b2);

	const Advection<PDEParameterOption::Constant> b((this -> b_vector_).data());

	return b;
}

template <typename ... Extensions>
Real PDE_Parameter_Functional< ... Extension>::eval_K(const Real& angle, const Real& intensity, const lambda::type<1>& lambda) const
{
	// Check for proper values of angle and intensity
	// Notice that we keep angle in [0.0, EIGEN_PI] exploiting the periodicity of the matrix K
	if (angle < 0.0 || angle > EIGEN_PI || intensity <= 0.0)
		return std::numeric_limits<Real>::infinity();
	
	else
	{
		carrier_ -> get_model() -> getRegressionData().setK(build_K(angle, intensity));

		MatrixXr z_hat = carrier_ -> apply(lambda); // apply or apply_to_b ??? VectorXr or MatrixXr?

		return ((carrier_ -> get_zp) - z_hat).squaredNorm();
    }
}

template <typename ... Extensions>
Real PDE_Parameter_Functional< ... Extension>::eval_b(const Real& b1, const Real& b2, const lambda::type<1>& lambda) const
{
	carrier_ -> get_model() -> getRegressionData().setBeta(build_b(b1, b2));

	MatrixXr z_hat = carrier_ -> apply(lambda); // apply or apply_to_b ???

	return ((carrier_ -> get_zp) - z_hat).squaredNorm();
    }
}

template <typename ... Extensions>
Real PDE_Parameter_Functional< ... Extension>::eval_c(const Real& c, const lambda::type<1>& lambda)) const
{
	carrier_ -> get_model() -> getRegressionData().setC(c);

	MatrixXr z_hat = carrier_ -> apply(lambda); // apply or apply_to_b ???

	return ((carrier_ -> get_zp) - z_hat).squaredNorm();
    }
}

template <typename ... Extensions>
VextorXr PDE_Parameter_Functional< ... Extension>::eval_grad_K(const Real& angle, const Real& intensity,
															   const lambda::type<1>& lambda), const Real& h) const
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

template <typename ... Extensions>
VectorXr PDE_Parameter_Functional< ... Extension>::eval_grad_b(const Real& b1, const Real& b2, 
															   const lambda::type<1>& lambda, const Real& h) const
{	
	VectorXr res;
	
	res << (eval_b(b1 + h, b2, lambda) - eval_b(b1 - h, b2, lambda)) / (2. * h),
		   (eval_b(b1, b2 + h, lambda) - eval_b(b1, b2 - h, lambda)) / (2. * h);
	
	return res;

}


template <typename ... Extensions>
Real PDE_Parameter_Functional< ... Extension>::eval_grad_c(const Real& c, const lambda::type<1>& lambda), const Real& h) const
{
	return (eval_c(c + h, lambda) - eval_c(c - h, lambda)) / (2. * h);
}


#endif
