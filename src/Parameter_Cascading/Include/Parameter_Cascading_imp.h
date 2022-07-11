#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

#include "../../Lambda_Optimization/Include/Optimization_Data.h"
#include "../../Lambda_Optimization/Include/Function_Variadic.h"
#include "../../Lambda_Optimization/Include/Newton.h"
#include "../../Lambda_Optimization/Include/Optimization_Methods_Factory.h"
#include "Optimization_Algorithm.h"

#include <memory>
#include <functional>
#include <limits>

template <UInt ORDER, UInt mydim, UInt ndim>
std::pair<Real, Real>
Parameter_Cascading<ORDER, mydim, ndim>::compute_GCV(Carrier<RegressionDataElliptic>& carrier,
													 GCV_Exact<Carrier<RegressionDataElliptic>, 1>& solver,
													 Real lambda_init) const
{	

	Function_Wrapper<Real, Real, Real, Real, GCV_Exact<Carrier<RegressionDataElliptic>, 1>> Fun(solver);
	const OptimizationData optr = H.getModel().getOptimizationData();

	std::unique_ptr<Opt_methods<Real,Real,GCV_Exact<Carrier<RegressionDataElliptic>, 1>>>
	optim_p = Opt_method_factory<Real, Real, GCV_Exact<Carrier<RegressionDataElliptic>, 1>>::create_Opt_method(optr.get_criterion(), Fun);

	// Start from 6 lambda and find the minimum value of GCV to pick a good initialization for Newton method
	std::vector<Real> lambda_grid = {5.000000e-05, 1.442700e-03, 4.162766e-02, 1.201124e+00, 3.465724e+01, 1.000000e+03};
	UInt dim = lambda_grid.size();
	Real lambda_min;
	Real GCV_min = -1.0;
	for (UInt i = 0; i < dim; i++)
	{
		Real evaluation = Fun.evaluate_f(lambda_grid[i]);
		if (evaluation < GCV_min || i == 0)
		{
			GCV_min = evaluation;
			lambda_min = lambda_grid[i];
		}
	}
	// If lambda_init <= 0, use the one from the grid
	if (lambda_init > lambda_min/4 || lambda_init <= 0)
		lambda_init = lambda_min/8;

	Checker ch;
	std::vector<Real> lambda_values;
	std::vector<Real> GCV_values;

	// Compute optimal lambda
	std::pair<Real, UInt> lambda_couple = 
	optim_p->compute(lambda_init, optr.get_stopping_criterion_tol(), 40, ch, GCV_values, lambda_values);

	// Return the GCV and the optimal lambda found
	return {lambda_couple.first, GCV_values[GCV_values.size()-1]};
}


template <UInt ORDER, UInt mydim, UInt ndim>
template <typename ParameterType>
ParameterType Parameter_Cascading<ORDER, mydim, ndim>::step(const ParameterType& init, const ParameterType& lower_bound, const ParameterType& upper_bound, 
	const ParameterType& periods, const std::function<Real (ParameterType, Real)>& F, const std::function<ParameterType (ParameterType, Real)>& dF, 
	const std::function<void (ParameterType)>& set_param)
{
	// Current solution and the previous solution for initialization
	ParameterType old_sol = init;
	ParameterType best_sol;

	// Variable to store the best iteration
	UInt best_iter;

	// Print initial solution
	Rprintf("Initial sol: ");
	printer<ParameterType>(old_sol);

	// Initialize GCV and variables to check if GCV is increasing
	GCV = -1.0;
	UInt counter_GCV_increasing = 0; // Variable to count how many iterations present an increasing GCV
	bool finer_grid = false;		 // Finer grid to activate when GCV is increasing
	Real old_GCV = std::numeric_limits<Real>::max();

	for (UInt iter = 0; iter < lambdas.size(); ++iter)
	{
		Rprintf("Finding optimal sol for lambda = %e\n", lambdas(iter));

		// Optimization step
		ParameterType init(old_sol); // Initialization done with previous solution as presented in \cite{Bernardi}
		ParameterType opt_sol;

		Real lambda = lambdas(iter);
		std::function<Real (ParameterType)> F_ = [&F, lambda](ParameterType x){return F(x, lambda);}; // Fix lambda in F
		
		if(optimization_algorithm == 0)
		{
			Parameter_Gradient_Descent_fd<ParameterType> param = {lower_bound, upper_bound, periods};
			std::function<ParameterType (ParameterType)> dF_ = [&dF, lambda](ParameterType x){return dF(x, lambda);}; // Fix lambda in dF
			Gradient_Descent_fd<ParameterType, Real> opt(F_, dF_, init, param);
			opt.apply();
			opt_sol = opt.get_solution();
		}
		else if(optimization_algorithm == 1)
		{
			Parameter_Genetic_Algorithm<ParameterType> param = {100, lower_bound, upper_bound};
			Genetic_Algorithm<ParameterType, Real> opt(F_, init, param);
			opt.apply();
			opt_sol = opt.get_solution();
		}

		// Compute GCV with the new parameters
		set_param(opt_sol);

		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(H.getModel().getRegressionData(), H.getModel(), H.getModel().getOptimizationData());
		GCV_Exact<Carrier<RegressionDataElliptic>, 1> solver(carrier);
		
		Rprintf("Computing GCV with the optimal sol for lambda = %e\n", lambdas(iter));

		std::pair<Real, Real> opt_sol_GCV = compute_GCV(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		
		if(iter == 0 || opt_sol_GCV.second <= GCV)
		{
			lambda_opt = opt_sol_GCV.first;
			GCV = opt_sol_GCV.second;
			best_sol = opt_sol;
			best_iter = iter;
		}

		if(iter == 0)
			old_GCV = GCV;
		
		Rprintf("Optimal sol for lambda = %e found\n", lambdas(iter));
		Rprintf("Optimal sol: ");
		printer<ParameterType>(opt_sol);

		// DEBUGGING
		Rprintf("GCV found: %f\n", opt_sol_GCV.second);
		Rprintf("Best GCV: %f\n", GCV);

		old_sol = opt_sol;
		
		// Check increasing GCV
		if(!finer_grid)
		{
			if(old_GCV < opt_sol_GCV.second)
			{
				counter_GCV_increasing++;

				// Build a finer grid of lambdas if GCV is increasing for 3 iterations in a row
				if(counter_GCV_increasing == 3)
				{
					Rprintf("Increasing GCV, restart Parameter Cascading algortihm with finer grid of lambdas\n");
					lambdas.resize(iter * 3, 1);
					lambdas = VectorXr::LinSpaced(iter * 3, lambdas(0), lambdas(best_iter + 1));
					iter = -1;
					counter_GCV_increasing = 0;
					old_sol = best_sol;
					finer_grid = true;
				}
			}
			else
				counter_GCV_increasing = 0;
		}
		old_GCV = opt_sol_GCV.second;

	}

	// Set the new parameter in RegressionData
	set_param(best_sol);

	Rprintf("Final optimal sol found\n");
	Rprintf("Optimal sol: ");
	printer<ParameterType>(best_sol);

	// DEBUGGING
	Rprintf("best iter = %d\n", best_iter);
	Rprintf("Final GCV: %f\n", GCV);
	Rprintf("Final optimal lambda for GCV: %e\n", lambda_opt);
	
	return best_sol;
}


template <UInt ORDER, UInt mydim, UInt ndim>
Output_Parameter_Cascading Parameter_Cascading<ORDER, mydim, ndim>::apply(void)
{
	Rprintf("Start Parameter_Cascading Algorithm\n");

	if(update_K)
	{	
		Rprintf("Finding diffusion matrix K\n");
		Eigen::Vector2d init(diffusion(0), diffusion(1));
		Eigen::Vector2d lower_bound(0.0, 0.0);
		Eigen::Vector2d upper_bound(EIGEN_PI, 1000.0);
		Eigen::Vector2d periods(EIGEN_PI, 0.0);
		std::function<Real (Eigen::Vector2d, Real)> F = [this](Eigen::Vector2d x, Real lambda){return this -> H.eval_K(x(0), x(1), lambda);};
		std::function<Eigen::Vector2d (Eigen::Vector2d, Real)> dF = [this](Eigen::Vector2d x, Real lambda){return this -> H.eval_grad_K(x(0),x(1), lambda);};
		std::function<void (Eigen::Vector2d)> set_param = [this](Eigen::Vector2d x){this -> H.set_K(x(0), x(1));};

		diffusion = step<Eigen::Vector2d>(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}

	if(update_alpha)
	{	
		Rprintf("Finding diffusion angle\n");
		Real init(diffusion(0));
		Real lower_bound(0.0);
		Real upper_bound(EIGEN_PI);
		Real periods(EIGEN_PI);
		std::function<Real (Real, Real)> F = [this](Real x, Real lambda){return this -> H.eval_K(x, this -> diffusion(1), lambda);};
		std::function<Real (Real, Real)> dF = [this](Real x, Real lambda){return this -> H.eval_grad_angle(x, this -> diffusion(1), lambda);};
		std::function<void (Real)> set_param = [this](Real x){this -> H.set_K(x, this -> diffusion(1));};

		diffusion(0) = step<Real>(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}	

	if(update_intensity)
	{	
		Rprintf("Finding diffusion intensity\n");
		Real init(diffusion(1));
		Real lower_bound(0.0);
		Real upper_bound(1000.0);
		Real periods(0.0);
		std::function<Real (Real, Real)> F = [this](Real x, Real lambda){return this -> H.eval_K(this -> diffusion(0), x, lambda);};
		std::function<Real (Real, Real)> dF = [this](Real x, Real lambda){return this -> H.eval_grad_intensity(this -> diffusion(0), x, lambda);};
		std::function<void (Real)> set_param = [this](Real x){this -> H.set_K(this -> diffusion(0), x);};

		diffusion(1) = step<Real>(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}	
	
	
	if(update_b)
	{
		Rprintf("Finding advection vector b\n");
		Eigen::Vector2d init(b(0), b(1));
		Eigen::Vector2d lower_bound(-1000.0, -1000.0);
		Eigen::Vector2d upper_bound(1000.0, 1000.0);
		Eigen::Vector2d periods(0.0, 0.0);
		std::function<Real (Eigen::Vector2d, Real)> F = [this](Eigen::Vector2d x, Real lambda){return this -> H.eval_b(x(0), x(1), lambda);};
		std::function<Eigen::Vector2d (Eigen::Vector2d, Real)> dF = [this](Eigen::Vector2d x, Real lambda){return this -> H.eval_grad_b(x(0),x(1), lambda);};
		std::function<void (Eigen::Vector2d)> set_param = [this](Eigen::Vector2d x){this -> H.set_b(x(0), x(1));};

		b = step<Eigen::Vector2d>(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}
			
	if(update_c)
	{
		Rprintf("Finding reaction coefficient c\n");
		Real init(c);
		Real lower_bound(-1000.0);
		Real upper_bound(1000.0);
		Real periods(0.0);
		std::function<Real (Real, Real)> F = [this](Real x, Real lambda){return this -> H.eval_c(x, lambda);};
		std::function<Real (Real, Real)> dF = [this](Real x, Real lambda){return this -> H.eval_grad_c(x, lambda);};
		std::function<void (Real)> set_param = [this](Real x){this -> H.set_c(x);};

		c = step<Real>(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}

	Eigen::Matrix2d K = H.compute_K(diffusion(0), diffusion(1));

	return {diffusion, K, b, c, lambda_opt};
}

#endif