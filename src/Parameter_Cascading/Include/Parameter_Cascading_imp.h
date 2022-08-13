#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

#include <memory>
#include <functional>
#include <limits>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <algorithm>

#include "../../Lambda_Optimization/Include/Optimization_Data.h"
#include "../../Lambda_Optimization/Include/Function_Variadic.h"
#include "../../Lambda_Optimization/Include/Newton.h"
#include "../../Lambda_Optimization/Include/Optimization_Methods_Factory.h"
#include "Optimization_Parameter_Cascading.h"
#include "../../C_Libraries/roptim.h"

using namespace roptim;

// Wrapper needed to use Roptim
class optimWrapper : public Functor
{
	public:
		optimWrapper(const std::function<Real (VectorXr)>& F_, const std::function<VectorXr (VectorXr)>& dF_)
		: F(F_), dF(dF_) {};

		Real operator()(const VectorXr &x) override
		{
			return F(x);
  		};

  		void Gradient(const VectorXr &x, VectorXr& grad) override
		{
			grad = dF(x);
  		};

private:
  	const std::function<Real (VectorXr)>& F;
  	const std::function<VectorXr (VectorXr)>& dF;
};


template <UInt ORDER, UInt mydim, UInt ndim>
template <typename EvaluationType>
std::pair<Real, Real>
Parameter_Cascading<ORDER, mydim, ndim>::compute_GCV(Carrier<RegressionDataElliptic>& carrier,
													 EvaluationType& solver,
													 Real lambda_init) const
{	

	Function_Wrapper<Real, Real, Real, Real, EvaluationType> Fun(solver);
	const OptimizationData optr = H.getModel().getOptimizationData();

	std::unique_ptr<Opt_methods<Real,Real,EvaluationType>>
	optim_p = Opt_method_factory<Real, Real, EvaluationType>::create_Opt_method(optr.get_criterion(), Fun);

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
VectorXr Parameter_Cascading<ORDER, mydim, ndim>::step(VectorXr init, const UInt& opt_algo, const std::function<Real (VectorXr, Real)>& F, 
	const std::function<VectorXr (VectorXr, Real)>& dF, const std::function<void (VectorXr)>& set_param)
{
	UInt dim = init.size();

	VectorXr lower_bound(dim);
	VectorXr upper_bound(dim);
	VectorXr periods(dim);
	for(UInt i = 0; i < dim; ++i)
	{
		lower_bound(i) = std::numeric_limits<Real>::min();
		upper_bound(i) = std::numeric_limits<Real>::max();
		periods(i) = 0.0;
	}

	return step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param);
} 

template <UInt ORDER, UInt mydim, UInt ndim>
VectorXr Parameter_Cascading<ORDER, mydim, ndim>::step(VectorXr init, const UInt& opt_algo, const VectorXr& lower_bound, const VectorXr& upper_bound, 
	const VectorXr& periods, const std::function<Real (VectorXr, Real)>& F, const std::function<VectorXr (VectorXr, Real)>& dF, 
	const std::function<void (VectorXr)>& set_param, bool constraint)
{
	// Current optimal solution and best solution
	VectorXr opt_sol = init;
	VectorXr best_sol(init.size());

	// Variable to store the best iteration
	UInt best_iter;

	// Variable to avoid boundary problems exploiting periodicity
	Real eps = 1e-3;

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
		std::function<VectorXr (VectorXr)> dF_ = [&dF, lambda](VectorXr x){return dF(x, lambda);}; // Fix lambda in dF

		if(opt_algo == 0) // L-BFGS-B
		{
			Rprintf("Start L-BFGS-B algortihm\n");
	   		optimWrapper function_to_optimize(F_, dF_);
  			Roptim<optimWrapper> solver("L-BFGS-B");
  			solver.set_lower(lower_bound);
  			solver.set_upper(upper_bound);
  			solver.minimize(function_to_optimize, opt_sol);

  			// Exploit periodicity
  			for(UInt i = 0; i < init.size(); ++i)
  			{
  				if(periods(i) != 0.0)
  				{
  					if(std::abs(opt_sol(i) - lower_bound(i)) < eps)
  					{
  						opt_sol(i) = upper_bound(i) - eps;
  						solver.minimize(function_to_optimize, opt_sol);
  					}
  					else if(std::abs(opt_sol(i) - upper_bound(i)) < eps)
  					{
  						opt_sol(i) = lower_bound(i) + eps;
  						solver.minimize(function_to_optimize, opt_sol);
  					}
				}  					
  			}

  			Rprintf("End L-BFGS-B algorithm\n");
		}
		else if(opt_algo == 1 && !constraint) // BFGS
		{
			Rprintf("Start BFGS algortihm\n");
	   		optimWrapper function_to_optimize(F_, dF_);
  			Roptim<optimWrapper> solver("BFGS");
  			solver.minimize(function_to_optimize, opt_sol);
  			Rprintf("End BFGS algorithm\n");
		}
		else if(opt_algo == 2 && !constraint) // CG
		{
			Rprintf("Start CG algortihm\n");
	   		optimWrapper function_to_optimize(F_, dF_);
  			Roptim<optimWrapper> solver("CG");
  			solver.minimize(function_to_optimize, opt_sol);
  			Rprintf("End CG algorithm\n");
		}
		else if(opt_algo == 3 && !constraint) // Nelder-Mead
		{
			Rprintf("Start Nelder-Mead algortihm\n");
	   		optimWrapper function_to_optimize(F_, dF_);
  			Roptim<optimWrapper> solver("Nelder-Mead");
  			solver.minimize(function_to_optimize, opt_sol);
  			Rprintf("End Nelder-Mead algorithm\n");
		}
		else if(opt_algo == 4) // Gradient
		{
			Parameter_Gradient_Descent_fd param = {lower_bound, upper_bound, periods};
			Gradient_Descent_fd opt(F_, dF_, init, param);
			opt.apply();
			opt_sol = opt.get_solution();
			init = opt_sol; // init modified to initialize the next iteration with the actual optimal solution
		}
		else if(opt_algo == 5) // Genetic
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
		// By default GCV_Exact is used; If user set GCV_Stochastic option in input, then it will be used
		set_param(opt_sol);
		std::pair<Real, Real> opt_sol_GCV;
		
		Rprintf("Computing GCV with the optimal sol for lambda = %e\n", lambdas(iter));

		OptimizationData& optr = H.getModel().getOptimizationData();
		Carrier<RegressionDataElliptic> carrier = CarrierBuilder<RegressionDataElliptic>::build_plain_carrier(H.getModel().getRegressionData(), H.getModel(), optr);
		if(optr.get_loss_function() == "GCV" && (optr.get_DOF_evaluation() == "stochastic" || optr.get_DOF_evaluation() == "not_required"))
		{
			GCV_Stochastic<Carrier<RegressionDataElliptic>, 1> solver(carrier, true);
			opt_sol_GCV = compute_GCV<GCV_Stochastic<Carrier<RegressionDataElliptic>, 1>>(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		}
		else
		{
			GCV_Exact<Carrier<RegressionDataElliptic>, 1> solver(carrier);
			opt_sol_GCV = compute_GCV<GCV_Exact<Carrier<RegressionDataElliptic>, 1>>(carrier, solver, lambda_opt); // Use the last optimal lambda found as initial lambda computing GCV
		}
		
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
					iter = lambdas.size();
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
	//Real eps = 1e-3;

	if(update_K)
	{	
		Rprintf("Finding diffusion matrix K\n");
		VectorXr init = diffusion;
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_diffusion_opt();

		UInt dim = (ndim == 2) ? 2 : 4;

		VectorXr lower_bound(dim);
		VectorXr upper_bound(dim);
		VectorXr periods(dim); // periods for each diffusion parameter (if variable not periodic then period = 0.0)

		if(ndim == 2)
		{
			lower_bound << 0.0, 0.0;
			upper_bound << EIGEN_PI, 1000.0;
			periods << EIGEN_PI, 0.0;
		}
		else if(ndim == 3)
		{
			lower_bound << 0.0, 0.0, 1.0, 1.0;
			upper_bound << EIGEN_PI, EIGEN_PI, 1000.0, 1000.0;
			periods << EIGEN_PI, EIGEN_PI, 0.0, 0.0;
		}
				
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda)
		{
			return this -> H.eval_K(x, lambda);
		};

		std::function<VectorXr (VectorXr, Real)> dF = [this, &lower_bound, &upper_bound](VectorXr x, Real lambda)
		{
			return this -> H.eval_grad_K(x, lower_bound, upper_bound, lambda);
		};

		std::function<void (VectorXr)> set_param = [this](VectorXr x)
		{
			this -> H.template set_K<VectorXr>(x);
		};

		diffusion = step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param, true);
	}

	if(update_K_direction) // Update only the diffusion angles parameter (the first in 2D case, the first two in 3D case)
	{	
		Rprintf("Finding K direction\n");

		UInt angle_dim = (ndim == 2) ? 1 : 2;

		VectorXr init(angle_dim);
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_diffusion_opt();
		VectorXr lower_bound(angle_dim);
		VectorXr upper_bound(angle_dim);
		VectorXr periods(angle_dim);

		std::function<Real (VectorXr, Real)> F;
		std::function<VectorXr (VectorXr, Real)> dF;
		std::function<void (VectorXr)> set_param;

		if(ndim == 2)
		{
			init << diffusion(0);
			lower_bound << 0.0;
			upper_bound << EIGEN_PI;
			periods << EIGEN_PI;

			F = [this](VectorXr x, Real lambda)
			{
				VectorXr param(2);
				param << x(0), this -> diffusion(1);	
				return this -> H.eval_K(param, lambda);
			};

			dF = [this, &upper_bound, &lower_bound](VectorXr x, Real lambda)
			{
				VectorXr param(2);
				VectorXr lb(2);
				VectorXr ub(2);
			
				param << x(0), this -> diffusion(1);
				lb << lower_bound, 0.0;
				ub << upper_bound, 1000.0;

				VectorXr grad(1);
				grad << this -> H.eval_grad_K(param, lb, ub, lambda)(0);
				return grad;
			};


			set_param = [this](VectorXr x)
			{
				VectorXr param(2);
				param << x(0), this -> diffusion(1);
				this -> H.template set_K<VectorXr>(param);
			};

			diffusion(0) = step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param, true)(0);
		}
		else if(ndim == 3)
		{
			init << diffusion(0), diffusion(1);
			lower_bound << 0.0, 0.0;
			upper_bound << EIGEN_PI, EIGEN_PI;
			periods << EIGEN_PI, EIGEN_PI;

			F = [this](VectorXr x, Real lambda)
			{
				VectorXr param(4);
				param << x(0), x(1), this -> diffusion(2), this -> diffusion(3);
				return this -> H.eval_K(param, lambda);
			};

			dF = [this, &upper_bound, &lower_bound](VectorXr x, Real lambda)
			{
				VectorXr param(4);
				VectorXr lb(4);
				VectorXr ub(4);
			
				param << x(0), x(1), this -> diffusion(2), this -> diffusion(3);
				lb << lower_bound, 1.0, 1.0;
				ub << upper_bound, 1000.0, 1000.0;

				VectorXr grad(2);
				VectorXr tmp(4);
				tmp = this -> H.eval_grad_K(param, lb, ub, lambda);
				grad << tmp(0), tmp(1);

				return grad;
			};

			set_param = [this](VectorXr x)
			{
				VectorXr param(4);
				param << x(0), x(1), this -> diffusion(2), this -> diffusion(3);
				this -> H.template set_K<VectorXr>(param);
			};

			VectorXr res = step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param, true);
			diffusion(0) = res(0);
			diffusion(1) = res(1);
		}
	}

	if(update_K_eigenval_ratio) // Update only the intensities in diffusion (the last parameter in 2D, the last two in 3D)
	{
		Rprintf("Finding K eigenval ratio\n");

		UInt intensity_dim = (ndim == 2) ? 1 : 2;

		VectorXr init(intensity_dim);
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_diffusion_opt();
		VectorXr lower_bound(intensity_dim);
		VectorXr upper_bound(intensity_dim);
		VectorXr periods(intensity_dim);

		std::function<Real (VectorXr, Real)> F;
		std::function<VectorXr (VectorXr, Real)> dF;
		std::function<void (VectorXr)> set_param;

		if(ndim == 2)
		{		
			init << diffusion(1);
			lower_bound << 0.0;
			upper_bound << 1000.0;
			periods << 0.0;

			F = [this](VectorXr x, Real lambda)
			{
				VectorXr param(2);
				param << this -> diffusion(0), x(0);
				return this -> H.eval_K(param, lambda);
			};

			dF = [this, &upper_bound, &lower_bound](VectorXr x, Real lambda)
			{
				VectorXr param(2);
				VectorXr lb(2);
				VectorXr ub(2);
			
				param << this -> diffusion(0), x(0);
				lb << 0.0, lower_bound;
				ub << EIGEN_PI, upper_bound;

				VectorXr grad(1);
				grad << this -> H.eval_grad_K(param, lb, ub, lambda)(1);
				return grad;
			};


			set_param = [this](VectorXr x)
			{
				VectorXr param(2);
				param << this -> diffusion(0), x(0);
				this -> H.template set_K<VectorXr>(param);
			};

			diffusion(1) = step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param, true)(0);
		}
		else if(ndim == 3)
		{
			init << diffusion(2), diffusion(3);
			lower_bound << 1.0, 1.0;
			upper_bound << 1000.0, 1000.0;
			periods << 0.0, 0.0;

			F = [this](VectorXr x, Real lambda)
			{
				VectorXr param(4);
				param << this -> diffusion(0), this -> diffusion(1), x(0), x(1);
				return this -> H.eval_K(param, lambda);
			};

			dF = [this, &upper_bound, &lower_bound](VectorXr x, Real lambda)
			{
				VectorXr param(4);
				VectorXr lb(4);
				VectorXr ub(4);
			
				param << this -> diffusion(0), this -> diffusion(1), x(0), x(1);
				lb << 0.0, 0.0, lower_bound;
				ub << EIGEN_PI, EIGEN_PI, upper_bound;

				VectorXr grad(2);
				VectorXr tmp(4);
				tmp = this -> H.eval_grad_K(param, lb, ub, lambda);
				grad << tmp(2), tmp(3);

				return grad;
			};

			set_param = [this](VectorXr x)
			{
				VectorXr param(4);
				param << this -> diffusion(0), this -> diffusion(1), x(0), x(1);
				this -> H.template set_K<VectorXr>(param);
			};

			VectorXr res = step(init, opt_algo, lower_bound, upper_bound, periods, F, dF, set_param, true);
			diffusion(2) = res(0);
			diffusion(3) = res(1);
		}
	}

	if(update_anisotropy_intensity)
	{
		Rprintf("Finding anisotropy intensity\n");
		VectorXr init(1);
		VectorXr lower_bound(1);
		VectorXr upper_bound(1);
		VectorXr periods(1);
		init << aniso_intensity;
		lower_bound << 0.0;
		upper_bound << 1000.0;
		periods << 0.0;
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_diffusion_opt();

		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda)
		{	
			MatrixXr new_K = K * x(0);
			return this -> H.eval_K(new_K, lambda);
		};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda)
		{	
			return this -> H.eval_grad_aniso_intensity(K, x(0), lambda);
		};
		std::function<void (VectorXr)> set_param = [this](VectorXr x)
		{	
			MatrixXr new_K = K * x(0);
			this -> H.template set_K<MatrixXr>(new_K);
		};
		
		aniso_intensity = step(init, opt_algo, F, dF, set_param)(0);
	}
	
	if(update_b)
	{
		Rprintf("Finding advection vector b\n");
		VectorXr init = b;
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_advection_opt();
				
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda)
		{
			return this -> H.eval_b(x, lambda);
		};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda)
		{
			return this -> H.eval_grad_b(x, lambda);
		};
		std::function<void (VectorXr)> set_param = [this](VectorXr x)
		{
			this -> H.set_b(x);
		};

		b = step(init, opt_algo, F, dF, set_param);
	}
			
	if(update_c)
	{
		Rprintf("Finding reaction coefficient c\n");
		VectorXr init(1);
		init << c;
		UInt opt_algo = H.getModel().getRegressionData().get_parameter_cascading_reaction_opt();
			
		std::function<Real (VectorXr, Real)> F = [this](VectorXr x, Real lambda)
		{
			return this -> H.eval_c(x(0), lambda);
		};
		std::function<VectorXr (VectorXr, Real)> dF = [this](VectorXr x, Real lambda)
		{
			return this -> H.eval_grad_c(x(0), lambda);
		};
		std::function<void (VectorXr)> set_param = [this](VectorXr x)
		{
			this -> H.set_c(x(0));
		};

		c = step(init, opt_algo, F, dF, set_param)(0);
	}

	Rprintf("End Parameter Cascading Algorithm\n");

	MatrixXr K_opt = H.getModel().getRegressionData().getK().template getDiffusionMatrix<ndim>();

	return {diffusion, aniso_intensity, K_opt, b, c, lambda_opt};
}

#endif
