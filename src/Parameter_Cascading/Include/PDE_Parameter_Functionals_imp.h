#ifndef __PDE_PARAMETER_FUNCTIONALS_IMP_H__
#define __PDE_PARAMETER_FUNCTIONALS_IMP_H__

#include <cmath> 
#include <limits>
#include <type_traits>
#include <utility>
#include "../../Regression/Include/Regression_Data.h"
#include "../../Lambda_Optimization/Include/Carrier.h"
#include "../../Lambda_Optimization/Include/Lambda_Optimizer.h"


template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_K(const VectorXr& DiffParam) const
{
	// Set the diffusion in RegressionData
	model.getRegressionData().getK().setDiffusion(DiffParam);

	// Recompute R1 matrix with new data
	// This allow to re-compute only the matrix dependent on PDE_parameters, avoiding to re-make the entire preapply() method
	model.template setR1<ORDER, mydim, ndim>(mesh);

	return;
}
			

template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_b(const VectorXr& AdvParam) const
{
	// Set the advection in RegressionData
	model.getRegressionData().getB().setAdvection(AdvParam);

	// Recompute R1 matrix with new data
	model.template setR1<ORDER, mydim, ndim>(mesh);

	return;
}


template <UInt ORDER, UInt mydim, UInt ndim>
void PDE_Parameter_Functional<ORDER, mydim, ndim>::set_c(const Real& c) const
{	
	// Set the reaction in RegressionData
	model.getRegressionData().getC().setReaction(c);

	// Recompute R1 matrix with new data
	model.template setR1<ORDER, mydim, ndim>(mesh);

	return;
}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_K(const VectorXr& DiffParam, const VectorXr& LowerBound, const VectorXr& UpperBound, const lambda::type<1>& lambda) const
{
	// Check for proper values of diffusion parameters
	// Usually angles are in [0.0, EIGEN_PI] and eigenvalue_ratios are positive
	for(UInt i = 0; i < DiffParam.size(); ++i)
	{
		if(DiffParam(i) < LowerBound(i) || DiffParam(i) > UpperBound(i))
			return std::numeric_limits<Real>::max();
	}
	
	// Set parameter in RegressionData
	set_K(DiffParam);
	
	// Solve the regression problem
	// Since this function will be called a lot of times in Parameter Cascading, GCV_stochastic is used (it's faster than GCV_Exact);
	// moreover it is good since it computes directly z_hat
	Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
	GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);
		
	solver.update_parameters(lambda); // solve the problem and compute z_hat

	VectorXr z_hat = solver.get_z_hat();
	VectorXr zp = *(model.getRegressionData().getObservations());
	Real res = (zp - z_hat).squaredNorm();

	return res;
}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_b(const VectorXr& AdvParam, const lambda::type<1>& lambda) const
{
	// Set parameter in RegressionData
	set_b(AdvParam);
	
	// Solve the regression problem
	// Since this function will be called a lot of times in Parameter Cascading, GCV_stochastic is used (it's faster than GCV_Exact);
	// moreover it is good since it computes directly z_hat
	Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
	GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);
		
	solver.update_parameters(lambda); // solve the problem and compute z_hat

	VectorXr z_hat = solver.get_z_hat();
	VectorXr zp = *(model.getRegressionData().getObservations());

	return (zp - z_hat).squaredNorm();
}


template <UInt ORDER, UInt mydim, UInt ndim>
Real PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_c(const Real& c, const lambda::type<1>& lambda) const
{
	// Set parameter in RegressionData
	set_c(c);
	
	// Solve the regression problem
	// Since this function will be called a lot of times in Parameter Cascading, GCV_stochastic is used (it's faster than GCV_Exact);
	// moreover it is good since it computes directly z_hat
	Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(model.getRegressionData(), model, model.getOptimizationData());
	GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);
		
	solver.update_parameters(lambda); // solve the problem and compute z_hat

	VectorXr z_hat = solver.get_z_hat();
	VectorXr zp = *(model.getRegressionData().getObservations());

	return (zp - z_hat).squaredNorm();
}


template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_K(const VectorXr& DiffParam, const VectorXr& LowerBound, const VectorXr& UpperBound, const lambda::type<1>& lambda, const Real& h) const
{
	UInt dim = (ndim == 2) ? 2 : 5;
	VectorXr res(dim);

	// Check if angle remain in a proper range after finite difference schemes
	VectorXr DiffParamLower(dim);
	VectorXr DiffParamUpper(dim);
	Real hLower;
	Real hUpper;

	for(UInt i = 0; i < dim; ++i)
	{	
		DiffParamLower = DiffParam;
		DiffParamUpper = DiffParam;

		if(DiffParam(i) - h < LowerBound(i))
			hLower = h;
		else
		{
			DiffParamLower(i) -= h;
			hLower = 2. * h;
		}

		if(DiffParam(i) + h > UpperBound(i))
			hUpper = h;
		else
		{
			DiffParamUpper(i) += h;
			hUpper = 2. * h;
		}

		res(i) = (eval_K(DiffParamUpper, LowerBound, UpperBound, lambda) - eval_K(DiffParamLower, LowerBound, UpperBound, lambda)) / std::min(hUpper,hLower);
	}
	
	return res;
}


template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_b(const VectorXr& AdvParam, const lambda::type<1>& lambda, const Real& h) const
{
	VectorXr res(ndim);

	for(UInt i = 0; i < ndim; ++i)
	{
		VectorXr bUpper = AdvParam;
		bUpper(i) += h;
		VectorXr bLower = AdvParam;
		bLower(i) -= h;

		res(i) = (eval_b(bUpper, lambda) - eval_b(bLower, lambda)) / (2. * h);
	}
		
	return res;

}


template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr PDE_Parameter_Functional<ORDER, mydim, ndim>::eval_grad_c(const Real& c, const lambda::type<1>& lambda, const Real& h) const
{
	VectorXr res(1);

	res(0) = (eval_c(c + h, lambda) - eval_c(c - h, lambda)) / (2. * h);

	return res;
}


#endif