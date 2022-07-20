#ifndef __OPTIMIZATION_PARAMETER_CASCADING_H__
#define __OPTIMIZATION_PARAMETER_CASCADING_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include <functional>
#include <vector>
#include <deque>
#include <type_traits>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

struct Parameter_Genetic_Algorithm
{
	unsigned int N;				// Population size
	VectorXr lower_bound;		// Lower bound for input values
	VectorXr upper_bound;		// Upper bound for output values
};

class Genetic_Algorithm
{
	private: typedef std::vector<VectorXr> populationType;
			 typedef std::vector<Real> evalType;

			 // Function to optimize
			 std::function<Real (VectorXr)> F; 

			 // Population of candidate solutions
			 populationType population;
			 
			 // Minimizer of F and minimal value of F
			 VectorXr best;
			 Real min_value;

			 // Genetic algortihm parameters
			 Parameter_Genetic_Algorithm param_genetic_algorithm;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if the best solution does not change for 3 iterations
			 bool goOn = true;

			 const unsigned int max_iterations_genetic_algorithm;

			 unsigned int seed;

			 // Generate a random VectorXr element from a truncated normal centered in "mean" 
			 // with standard deviation sigma
			 VectorXr get_random_element(const VectorXr& mean, const Real& sigma);

			 // Cumulative distribution function of a standard normal (phi function)
			 Real normal_cdf(const Real& x) const;

			 // Utility to compute the probit function, i.e. the quantile function of a standard normal
			 Real probit(const Real& u) const;

			 // Initialization step
			 void initialization(void);

			 // Evaluation step
			 evalType evaluation(void) const;

			 // Selection and Variation steps
			 void selection_and_variation(evalType values, Real alpha);
			 

	public: // Constructors
			Genetic_Algorithm(const std::function<Real (VectorXr)>& F_, const VectorXr& init, 
			const Parameter_Genetic_Algorithm& param_genetic_algorithm_,
			const unsigned int& max_iterations_genetic_algorithm_)
			: F(F_), best(init), param_genetic_algorithm(param_genetic_algorithm_), 
			max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_)
			 {
			 	population.resize(param_genetic_algorithm.N);
				population[0] = init;
				min_value = F(init);

				seed = std::chrono::system_clock::now().time_since_epoch().count();
			 };

			Genetic_Algorithm(const std::function<Real (VectorXr)>& F_, const VectorXr& init, const Parameter_Genetic_Algorithm& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 50u) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline VectorXr get_solution(void) const {return best;};
};


struct Parameter_Gradient_Descent_fd
{
	VectorXr lower_bound;	// Lower bound for input values
	VectorXr upper_bound;	// Upper bound for output values
	VectorXr periods;		// Periodicity for each input component (if periods[i] = 0.0 then no periodicity for i-th component)	
};


class Gradient_Descent_fd
{
	private: // Function to optimize
			 std::function<Real (VectorXr)> F;

			 // Approx of the gradient
			 std::function<VectorXr (VectorXr)> dF;

			 // Minimizer of F and minimal value of F
			 VectorXr best;

			 // Gradient Descent parameters
			 Parameter_Gradient_Descent_fd param_gradient_descent_fd;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if  increment < tolerance 
			 bool goOn = true;

			 const unsigned int max_iterations_gradient_descent_fd;
			 const Real tol_gradient_descent_fd;

			 void upgrade_best(void);
			 
	public: // Constructors
			Gradient_Descent_fd(const std::function<Real (VectorXr)>& F_, const std::function<VectorXr (VectorXr)>& dF_, 
			const VectorXr& init, const Parameter_Gradient_Descent_fd& param_gradient_descent_fd_,
			const unsigned int& max_iterations_gradient_descent_fd_,
			const Real tol_gradient_descent_fd_)
			: F(F_), dF(dF_), best(init), param_gradient_descent_fd(param_gradient_descent_fd_),
			max_iterations_gradient_descent_fd(max_iterations_gradient_descent_fd_),
			tol_gradient_descent_fd(tol_gradient_descent_fd_)
			{
			 	best = init;
			};

			Gradient_Descent_fd(const std::function<Real (VectorXr)>& F_, const std::function<VectorXr (VectorXr)>& dF_, const VectorXr& init,
				const Parameter_Gradient_Descent_fd& param_gradient_descent_fd_)
			: Gradient_Descent_fd(F_, dF_, init, param_gradient_descent_fd_, 50u, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline VectorXr get_solution(void) const {return best;};
};

#endif