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
													 GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>& solver,
													 Real lambda_init) const
{	

	Function_Wrapper<Real, Real, Real, Real, GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>> Fun(solver);
	const OptimizationData optr = H.getModel().getOptimizationData();

	// Stochastic computation of the GCV will be used with Newton_fd to be faster
	std::unique_ptr<Opt_methods<Real,Real,GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>>>
	optim_p = Opt_method_factory<Real, Real, GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>>::create_Opt_method(optr.get_criterion(), Fun);

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
void Parameter_Cascading<ORDER, mydim, ndim>::step_K(void)
{
	// Current solution and the previous solution for initialization
	Real old_angle = angle;
	Real old_intensity = intensity;
	Real new_angle;
	Real new_intensity;

	Rprintf("Initial angle and intensity: %f , %f\n", old_angle, old_intensity);

	// GCV value
	Real GCV_value = -1.0;

	// Parameters for optimization algorithm
	Eigen::Vector2d lower_bound(0.0, 0.0);
	Eigen::Vector2d upper_bound(EIGEN_PI, std::numeric_limits<Real>::max());
	Parameter_Genetic_Algorithm<Eigen::Vector2d> param = {100, lower_bound, upper_bound};

	UInt best_iter; // debugging purpose
	
	for (UInt iter = 0; iter < lambdas.size(); ++iter)
	{
		Rprintf("Finding optimal K for lambda = %e\n", lambdas(iter));

		// Function to optimize
		auto F = [this, &iter](Eigen::Vector2d x){return this -> H.eval_K(x(0),x(1), this -> lambdas(iter));};

		// Optimization step
		Eigen::Vector2d init(old_angle, old_intensity); // Initialization done with previous solution as presented in \cite{Bernardi}

		Genetic_Algorithm<Eigen::Vector2d, Real> opt(F, init, param);
		opt.apply();
		
		// Store the optimal solution
		Eigen::Vector2d opt_sol = opt.get_solution();
		new_angle = opt_sol(0);
		new_intensity = opt_sol(1);

		// Compute GCV with the new parameters
		H.set_K(new_angle, new_intensity);

		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(H.getModel().getRegressionData(), H.getModel(), H.getModel().getOptimizationData());
		GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);  // GCV_Stochastic is used to be faster with computations
		
		Rprintf("Computing GCV with the optimal K for lambda = %e\n", lambdas(iter));

		std::pair<Real, Real> opt_sol_GCV = compute_GCV(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		
		old_angle = new_angle;
		old_intensity = new_intensity;

		if(iter == 0 || opt_sol_GCV.first <= GCV_value)
		{
			lambda_opt = opt_sol_GCV.first;
			GCV_value = opt_sol_GCV.second;
			angle = new_angle;
			intensity = new_intensity;

			best_iter = iter;
		}

		Rprintf("Optimal K for lambda = %e found\n", lambdas(iter));
		Rprintf("Optimal angle and intensity: %f, %f\n", new_angle, new_intensity);
		
		++iter;
	}

	// Set the new parameter in RegressionData
	H.set_K(angle, intensity);

	Rprintf("Final optimal K found\n");
	Rprintf("Final optimal angle and intensity: %f, %f\n", angle, intensity);

	// DEBUGGING
	Rprintf("best iter = %d\n", best_iter);
	Rprintf("Final GCV: %f\n", GCV_value);
	Rprintf("Final optimal lambda for GCV: %e\n", lambda_opt);
	
	return;
}

template <UInt ORDER, UInt mydim, UInt ndim>
void Parameter_Cascading<ORDER, mydim, ndim>::step_b(void)
{
	// Current solution in the loop and the previous solution for initialization
	Real old_b1 = b1;
	Real old_b2 = b2;
	Real new_b1;
	Real new_b2;

	Rprintf("Initial b: %f , %f\n", old_b1, old_b2);

	// GCV value
	Real GCV_value = -1.0;

	// Parameters for optimization algorithm
	Eigen::Vector2d lower_bound(std::numeric_limits<Real>::min(), std::numeric_limits<Real>::min());
	Eigen::Vector2d upper_bound(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
	Parameter_Genetic_Algorithm<Eigen::Vector2d> param = {100, lower_bound, upper_bound};

	for (UInt iter = 0; iter < lambdas.size(); ++iter)
	{
		Rprintf("Finding optimal b for lambda = %e\n", lambdas(iter));

		// Function to optimize
		auto F = [this, &iter](Eigen::Vector2d x){return this -> H.eval_b(x(0),x(1), this -> lambdas(iter));};
		
		// Optimization step
		Eigen::Vector2d init(old_b1, old_b2); // Initialization done with previous solution as presented in \cite{Bernardi}

		Genetic_Algorithm<Eigen::Vector2d, Real> opt(F, init, param);
		opt.apply();
		
		// Store the optimal solution
		Eigen::Vector2d opt_sol = opt.get_solution();
		new_b1 = opt_sol(0);
		new_b2 = opt_sol(1);
		
		// Compute GCV with the new parameters
		H.set_b(new_b1, new_b2);

		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(H.getModel().getRegressionData(), H.getModel(), H.getModel().getOptimizationData());
		GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);  // GCV_Stochastic is used to be faster with computations

		Rprintf("Computing GCV with the optimal b for lambda = %e\n", lambdas(iter));

		std::pair<Real, Real> opt_sol_GCV = compute_GCV(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		
		old_b1 = new_b1;
		old_b2 = new_b2;

		if(iter == 0 || opt_sol_GCV.first <= GCV_value)
		{
			lambda_opt = opt_sol_GCV.first;
			GCV_value = opt_sol_GCV.second;
			b1= new_b1;
			b2 = new_b2;
		}

		Rprintf("Optimal b for lambda = %e found\n", lambdas(iter));
		Rprintf("Optimal b: %f, %f\n", new_b1, new_b2);
			
		++iter;
	}

	// Set the new parameter in RegressionData
	H.set_b(b1, b2);

	Rprintf("Final optimal b found\n");
	Rprintf("Final optimal b: %f, %f\n", b1, b2);

	// DEBUGGING
	Rprintf("Final GCV: %f\n", GCV_value);
	Rprintf("Final optimal lambda for GCV: %e\n", lambda_opt);
	
	
	return;
}

template <UInt ORDER, UInt mydim, UInt ndim>
void Parameter_Cascading<ORDER, mydim, ndim>::step_c(void)
{
	Rprintf("Finding optimal c for lambda = %e\n", lambdas(iter));

	// Current solution in the loop and the previous solution for initialization
	Real old_c = c;
	Real new_c;

	Rprintf("Initial c: %f\n", old_c);

	// GCV value
	Real GCV_value = -1.0;

	// Parameters for optimization algorithm
	Real lower_bound(std::numeric_limits<Real>::min());
	Real upper_bound(std::numeric_limits<Real>::max());
	Parameter_Genetic_Algorithm<Real> param = {100, lower_bound, upper_bound};

	for (UInt iter = 0; iter < lambdas.size(); ++iter)
	{
		// Function to optimize
		auto F = [this, &iter](Real x){return this -> H.eval_c(x, this -> lambdas(iter));};
		
		// Optimization step
		Real init{old_c}; // Initialization done with previous solution as presented in \cite{Bernardi}

		Genetic_Algorithm<Real, Real> opt(F, init, param);
		opt.apply();

		// Store the optimal solution
		Real new_c = opt.get_solution();
				
		// Compute GCV with the new parameters
		H.set_c(new_c);

		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(H.getModel().getRegressionData(), H.getModel(), H.getModel().getOptimizationData());
		GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);  // GCV_Stochastic is used to be faster with computations

		Rprintf("Computing GCV with the optimal c for lambda = %e\n", lambdas(iter));

		std::pair<Real, Real> opt_sol_GCV = compute_GCV(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		
		old_c = new_c;

		if(iter == 0 || opt_sol_GCV.first <= GCV_value)
		{
			lambda_opt = opt_sol_GCV.first;
			GCV_value = opt_sol_GCV.second;
			c = new_c;
		}
		
		Rprintf("Optimal c for lambda = %e found\n", lambdas(iter));
		Rprintf("Optimal c: %f\n", c);

		++iter;
		
	}

	// Set the new parameter in RegressionData
	H.set_c(c);
	
	Rprintf("Final optimal c found\n");
	Rprintf("Final optimal c: %f\n", c);

	// DEBUGGING
	Rprintf("Final GCV: %f\n", GCV_value);
	Rprintf("Final optimal lambda for GCV: %e\n", lambda_opt);

	return;

}


template <UInt ORDER, UInt mydim, UInt ndim>
Real Parameter_Cascading<ORDER, mydim, ndim>::apply(void)
{
	if(update_K)
	{	
		Rprintf("Finding diffusion matrix K\n");		
		step_K();
	}
	
	if(update_b)
	{
		Rprintf("Finding advection vector b\n");
		step_b();
	}
			
	if(update_c)
	{
		Rprintf("Finding reaction coefficient c\n");
		step_c();
	}

	return lambda_opt;
}

#endif