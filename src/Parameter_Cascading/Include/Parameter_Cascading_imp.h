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

template <typename InputCarrier>
Real Parameter_Cascading<InputCarrier>::compute_optimal_lambda(void) const
{
	Function_Wrapper<Real, Real, Real, Real, GCV_Stochastic<InputCarrier, 1>> Fun(H.get_solver());

	const OptimizationData * optr = H.get_solver().get_carrier().get_opt_data();

	std::unique_ptr<Opt_methods<Real,Real,GCV_Stochastic<InputCarrier, 1>>>
	optim_p = Opt_method_factory<Real, Real, GCV_Stochastic<InputCarrier, 1>>::create_Opt_method(optr->get_criterion(), Fun);

	// Choose initial lambda
	Real lambda_init = optr->get_initial_lambda_S();

	// Start from 6 lambda and find the minimum value of GCV to pick a good initialization for Newton method
	std::vector<Real> lambda_grid = {5.000000e-05, 1.442700e-03, 4.162766e-02, 1.201124e+00, 3.465724e+01, 1.000000e+03};
	UInt dim = lambda_grid.size();
	Real lambda_min;
	Real GCV_min = -1.0;
	for (UInt i = 0; i < dim; i++)
	{
		Real evaluation = Fun.evaluate_f(lambda_grid[i]); //only scalar functions;
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
	optim_p->compute(lambda_init, optr->get_stopping_criterion_tol(), 40, ch, GCV_values, lambda_values);

	return lambda_couple.first;
}


template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_K(void)
{
	// Vectors to store the optimal values for each lambda in lambdas
	VectorXr angles(lambdas.size() + 1);
	VectorXr intensities(lambdas.size() + 1);

	// Initialization
	angles(0) = angle;
	intensities(0) = intensity;

	// Vectors to store the GCV values for each lambda in lambdas
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		std::function<Real (Eigen::Vector2d)> F[&H, &lambdas, &iter](Eigen::Vector2d x){return H.eval_K(x(0),x(1), lamdas(iter))};
		Eigen::Vector2d init(angles(iter), intensities(iter));
		Eigen::Vector2d lower_bound(0.0, 0.0);
		Eigen::Vector2d upper_bound(EIGEN_PI, std::numeric_limits<Real>::infinity());

		Genetic_Algorithm<Eigen::Vector2d, Real> opt(F, init, {100, lower_bound, upper_bound});
		
		// Store the optimal solution
		Eigen::Vector2d opt_sol = opt.get_solution();
		angles(iter + 1) = opt_sol(0);
		intensities(iter + 1) = opt_sol(1);

		// Compute GCV with the new parameters
		H.set_K(angles(iter + 1), intensities(iter + 1));
		Real lambda_opt = compute_lambda_optimal();
		H.get_solver().update_parameters(lambda_opt);
		GCV_values(iter) = H.get_solver().compute_f(lambda_opt);
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.minCoeff(&min_GCV_pos);

	angle = angles(min_GCV_pos + 1); // GCV_values is shorter than angles due to initialization => index shifted
	intensity = intensities(min_GCV_pos + 1); // GCV_values is shorter than intensities due to initialization => index shifted

	H.set_K(angle, intensity);
	
	// Compute increment
	increment += std::sqrt((angle - anlges(0))*(angle - anlges(0)));
	increment += std::sqrt((intensity - intensities(0))*(intensity - intensities(0)));
	
	return;
}

template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_b(void)
{
	// Vectors to store the optimal values for each lambda in lambdas
	VectorXr b1_values(lambdas.size() + 1);
	VectorXr b2_values(lambdas.size() + 1);

	// Initialization
	b1_values(0) = b1;
	b2_values(0) = b2;

	// Vectors to store the GCV values for each lambda in lambdas_
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		std::function<Real (Eigen::Vector2d)> F[&H, &lambdas, &iter](Eigen::Vector2d x){return H.eval_b(x(0),x(1), lamdas(iter))};
		Eigen::Vector2d init(b1_values(iter), b2_values(iter));
		Eigen::Vector2d lower_bound(-std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity());
		Eigen::Vector2d upper_bound(std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity());

		Genetic_Algorithm<Eigen::Vector2d, Real> opt(F, init, {100, lower_bound, upper_bound});
		
		// Store the optimal solution
		Eigen::Vector2d opt_sol = opt.get_solution();
		b1_values(iter + 1) = opt_sol(0);
		b2_values(iter + 1) = opt_sol(1);
		
		// Compute GCV with the new parameters
		H.set_b(b1_values(iter + 1), b2_values(iter + 1));
		Real lambda_opt = compute_lambda_optimal();
		H.get_solver().update_parameters(lambda_opt);
		GCV_values(iter) = H.get_solver().compute_f(lambda_opt);
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.minCoeff(&min_GCV_pos);

	b1 = b1_values(min_GCV_pos + 1); // GCV_values is shorter than b1_values due to initialization => index shifted
	b2 = b2_values(min_GCV_pos + 1); // GCV_values is shorter than b2_values due to initialization => index shifted

	H.set_b(b1, b2);
	
	// Compute increment
	increment += std::sqrt((b1 - b1_values(0))*(b1 - b1_values(0)));
	increment += std::sqrt((b2 - b2_values(0))*(b2 - b2_values(0)));
	
	return;
}

template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_c(void)
{
	// Vector to store the optimal values for each lambda in lambdas
	VectorXr c_values(lambdas.size() + 1);

	// Initialization
	c_values(0) = c;

	// Vectors to store the GCV values for each lambda in lambdas
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		std::function<Real (Real)> F[&H, &lambdas, &iter](Real x){return H.eval_c(x, lamdas(iter))};
		Real init{c_values(iter)};
		Real lower_bound(-std::numeric_limits<Real>::infinity());
		Real upper_bound(std::numeric_limits<Real>::infinity());

		Genetic_Algorithm<Real, Real> opt(F, init, {100, lower_bound, upper_bound});
		
		// Store the optimal solution
		Real opt_sol = opt.get_solution();
		c_values(iter + 1) = opt_sol;
		
		// Compute GCV with the new parameters
		H.set_c(c_values(iter + 1));
		Real lambda_opt = compute_lambda_optimal();
		H.get_solver().update_parameters(lambda_opt);
		GCV_values(iter) = H.get_solver().compute_f(lambda_opt);
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.minCoeff(&min_GCV_pos);

	c = c_values(min_GCV_pos + 1); // GCV_values is shorter than c_values due to initialization => index shifted
	
	H.set_c(c);
	
	// Compute increment
	increment += std::sqrt(c - c_values(0));
	
	return;

}


template <typename InputCarrier>
bool Parameter_Cascading<InputCarrier>::apply(void)
{
	unsigned int iter = 0u;

	while(iter < max_iter_parameter_cascading && goOn)
	{	
		++iter;
		
		increment = 0.0;
		
		if(update_K)
			step_K();
		
		if(update_b)
			step_b();
			
		if(update_c)
			step_c();
		
		// Stop the procedure when all the updaters are false or when just one updater is true
		if(update_K*update_b == false && update_K*update_c == false && update_b*update_c == false)
			goOn = false;

		else
			goOn = increment > tol_parameter_cascading;
		
	}
	
	return (iter < max_iter_parameter_cascading);

}

#endif