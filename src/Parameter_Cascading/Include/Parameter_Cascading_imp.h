#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

#include "../../Lambda_Optimization/Include/Optimization_Data.h"
#include "../../Lambda_Optimization/Include/Function_Variadic.h"
#include "../../Lambda_Optimization/Include/Newton.h"
#include "../../Lambda_Optimization/Include/Optimization_Methods_Factory.h"
#include "Optimization_Parameter_Cascading.h"
#include "../../C_Libraries/L-BFGS-B/LBFGSB.h" 

#include <memory>
#include <functional>
#include <limits>
#include <iostream> 

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
VectorXr Parameter_Cascading<ORDER, mydim, ndim>::step(VectorXr init, const VectorXr& lower_bound, const VectorXr& upper_bound, 
	const VectorXr& periods, const std::function<Real (VectorXr, Real)>& F, const std::function<VectorXr (VectorXr, Real)>& dF, 
	const std::function<void (VectorXr)>& set_param)
{
	// Current optimal solution and best solution
	VectorXr opt_sol = init;
	VectorXr best_sol(init.size());

	// Variable to store the best iteration
	UInt best_iter;

	// Print initial solution
	Rprintf("Initial sol: ");
	for(unsigned int i = 0u; i < init.size(); ++i)
		Rprintf("%f ", init[i]);
	Rprintf("\n");

	// Initialize GCV and variables to check if GCV is increasing
	GCV = -1.0;
	UInt counter_GCV_increasing = 0; // Variable to count how many iterations present an increasing GCV
	bool finer_grid = false;		 // Finer grid to activate when GCV is increasing
	Real old_GCV = std::numeric_limits<Real>::max();

	for (UInt iter = 0; iter < lambdas.size(); ++iter)
	{
		Rprintf("Finding optimal sol for lambda = %e\n", lambdas(iter));

		Real lambda = lambdas(iter);

		std::function<Real (VectorXr)> F_ = [&F, lambda](VectorXr x){return F(x, lambda);}; // Fix lambda in F
		
		if(optimization_algorithm == 0) // L-BFGS-B
		{
			LBFGSBParam<Real> param;
			LBFGSBSolver<Real> solver(param);
			std::function<VectorXr (VectorXr)> dF_ = [&dF, lambda](VectorXr x){return dF(x, lambda);}; // Fix lambda in dF
			Real fx; // f(x) value

	   		UInt niter = solver.minimize(F_, dF_, opt_sol, fx, lower_bound, upper_bound); // opt_sol and fx will be directly modified
       		
       		// DEBUGGING
       		Rprintf("number of iter: %d\n", niter);
       		Rprintf("f(x): %f\n", fx);
       		
		}
		else if(optimization_algorithm == 1) // gradient
		{
			Parameter_Gradient_Descent_fd param = {lower_bound, upper_bound, periods};
			std::function<VectorXr (VectorXr)> dF_ = [&dF, lambda](VectorXr x){return dF(x, lambda);}; // Fix lambda in dF
			
			Gradient_Descent_fd opt(F_, dF_, init, param);
			opt.apply();
			
			opt_sol = opt.get_solution();
			init = opt_sol; // init modified to initialize the next iteration with the actual optimal solution
		}
		else if(optimization_algorithm == 2) // genetic
		{
			Parameter_Genetic_Algorithm param = {100, lower_bound, upper_bound};
			
			Genetic_Algorithm opt(F_, init, param);
			opt.apply();
			
			opt_sol = opt.get_solution();
			init = opt_sol; // init modified to initialize the next iteration with the actual optimal solution
		}
		
		Rprintf("Optimal sol for lambda = %e found\n", lambdas(iter));
		Rprintf("Optimal sol: ");
		for(unsigned int i = 0u; i < opt_sol.size(); ++i)
			Rprintf("%f ", opt_sol[i]);
		Rprintf("\n");

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

		// DEBUGGING
		Rprintf("GCV found: %f\n", opt_sol_GCV.second);
		Rprintf("Best GCV: %f\n", GCV);
		
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
					UInt start_finer_grid = (best_iter < 2) ? 0 : best_iter - 1;
					lambdas = VectorXr::LinSpaced(6, lambdas(start_finer_grid), lambdas(best_iter + 1));
					lambdas.resize(6, 1);
					iter = 0;
					init = best_sol;
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
	for(unsigned int i = 0u; i < best_sol.size(); ++i)
		Rprintf("%f ", best_sol[i]);
	Rprintf("\n");

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
		VectorXr init(2);
		init << diffusion(0), diffusion(1);
		VectorXr lower_bound(2);
		lower_bound << 0.0, 0.0;
		VectorXr upper_bound(2);
		upper_bound << EIGEN_PI, 1000.0;
		VectorXr periods(2);
		periods << EIGEN_PI, 0.0;
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda){return this -> H.eval_K(x(0), x(1), lambda);};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda){return this -> H.eval_grad_K(x(0),x(1), lambda);};
		std::function<void (VectorXr)> set_param = [this](VectorXr x){this -> H.set_K(x(0), x(1));};

		diffusion = step(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}

	if(update_K_main_direction)
	{	
		Rprintf("Finding K main direction\n");
		VectorXr init(1);
		init << diffusion(0);
		VectorXr lower_bound(1);
		lower_bound << 0.0;
		VectorXr upper_bound(1);
		upper_bound << EIGEN_PI;
		VectorXr periods(1);
		periods << EIGEN_PI;
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda){return this -> H.eval_K(x(0), this -> diffusion(1), lambda);};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda){return this -> H.eval_grad_K_main_direction(x(0), this -> diffusion(1), lambda);};
		std::function<void (VectorXr)> set_param = [this](VectorXr x){this -> H.set_K(x(0), this -> diffusion(1));};

		diffusion(0) = step(init, lower_bound, upper_bound, periods, F, dF, set_param)(0);
	}	

	if(update_K_eigenval_ratio)
	{	
		Rprintf("Finding K eigenval ratio\n");
		VectorXr init(1);
		init << diffusion(1);
		VectorXr lower_bound(1);
		lower_bound << 0.0;
		VectorXr upper_bound(1);
		upper_bound << 1000.0;
		VectorXr periods(1);
		periods << 0.0;
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda){return this -> H.eval_K(this -> diffusion(0), x(0), lambda);};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda){return this -> H.eval_grad_K_eigenval_ratio(this -> diffusion(0), x(0), lambda);};
		std::function<void (VectorXr)> set_param = [this](VectorXr x){this -> H.set_K(this -> diffusion(0), x(0));};

		diffusion(1) = step(init, lower_bound, upper_bound, periods, F, dF, set_param)(0);
	}	
	
	
	if(update_b)
	{
		Rprintf("Finding advection vector b\n");
		VectorXr init(2);
		init << b(0), b(1);
		VectorXr lower_bound(2);
		lower_bound << -1000.0, -1000.0;
		VectorXr upper_bound(2);
		upper_bound << 1000.0, 1000.0;
		VectorXr periods(2);
		periods << 0.0, 0.0;
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda){return this -> H.eval_b(x(0), x(1), lambda);};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda){return this -> H.eval_grad_b(x(0),x(1), lambda);};
		std::function<void (VectorXr)> set_param = [this](VectorXr x){this -> H.set_b(x(0), x(1));};

		b = step(init, lower_bound, upper_bound, periods, F, dF, set_param);
	}
			
	if(update_c)
	{
		Rprintf("Finding reaction coefficient c\n");
		VectorXr init(1);
		init << c;
		VectorXr lower_bound(1);
		lower_bound << -1000.0;
		VectorXr upper_bound(1);
		upper_bound << 1000.0;
		VectorXr periods(1);
		periods << 0.0;
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda){return this -> H.eval_c(x(0), lambda);};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda){return this -> H.eval_grad_c(x(0), lambda);};
		std::function<void (VectorXr)> set_param = [this](VectorXr x){this -> H.set_c(x(0));};

		c = step(init, lower_bound, upper_bound, periods, F, dF, set_param)(0);
	}

	MatrixXr K = H.compute_K(diffusion(0), diffusion(1));

	return {diffusion, K, b, c, lambda_opt};
}

#endif