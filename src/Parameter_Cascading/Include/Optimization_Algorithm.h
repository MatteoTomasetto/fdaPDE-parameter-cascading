#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

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

template <class DType>
struct Parameter_Genetic_Algorithm
{
	unsigned int N;			// Population size
	DType lower_bound;		// Lower bound for input values
	DType upper_bound;		// Upper bound for output values
};

// DType = Domain variable type, CType = Codomain variable type;
// DType can have one or more components, instead CType can be only a scalar type (1 component only)
template <class DType, class CType>
class Genetic_Algorithm
{
	private: typedef std::vector<DType> VectorXdtype;
			 typedef std::vector<CType> VectorXctype;

			 // Function to optimize
			 std::function<CType (DType)> F; 

			 // Population of candidate solutions
			 VectorXdtype population;
			 
			 // Minimizer of F and minimal value of F
			 DType best;
			 CType min_value;

			 // Genetic algortihm parameters
			 Parameter_Genetic_Algorithm<DType> param_genetic_algorithm;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if the best solution does not change for 3 iterations
			 bool goOn = true;

			 const unsigned int max_iterations_genetic_algorithm;

			 unsigned int seed;

			 // Generate a random DType element from a truncated normal centered in "mean" 
			 // with standard deviation sigma
			 // We need in-class definition to perform SFINAE properly (different code for vectorial and scalar cases)
			 // DType = Vector case (e.g. std::vector<double>, Eigen::Vector2d,...)
			 template <class SFINAE = DType>
			 const typename std::enable_if< !std::is_floating_point<SFINAE>::value, DType>::type
			 get_random_element(const DType& mean, const Real& sigma)
			 {
			 	std::default_random_engine generator(seed++);
			 	
			 	unsigned int ElemSize = mean.size();
			 	DType res;
			 	res.resize(ElemSize);

			 	typename DType::value_type sigma_converted = static_cast<typename DType::value_type>(sigma);

			 	// Loop over each component of mean and res and generate random the components of res
			 	for(unsigned int j = 0u; j < ElemSize; ++j)
			 	{	
			 		// Generate the j-th random component from a normal truncated wrt lower and upper bound in param_genetic_algorithm
			 		std::uniform_real_distribution<Real>
			 		unif_distr(normal_cdf(static_cast<Real>((param_genetic_algorithm.lower_bound[j] - mean[j])/sigma_converted)),
			 				   normal_cdf(static_cast<Real>((param_genetic_algorithm.upper_bound[j] - mean[j])/sigma_converted)));

			 		res[j] = mean[j] + sigma_converted * static_cast<typename DType::value_type>(probit(unif_distr(generator)));
			 	}
	
			 	return res;
			 }
			 // DType = Scalar case
			 template <class SFINAE = DType>
			 const typename std::enable_if< std::is_floating_point<SFINAE>::value, DType>::type
			 get_random_element(const DType& mean, const Real& sigma)
			 {

				DType sigma_converted = static_cast<DType>(sigma);
		 	
			 	// Generate the a number from a normal truncated in (a,b)
			 	std::default_random_engine generator(seed++);
			 	std::uniform_real_distribution<Real>
			 	unif_distr(normal_cdf(static_cast<Real>((param_genetic_algorithm.lower_bound - mean)/sigma_converted)),
			 			   normal_cdf(static_cast<Real>((param_genetic_algorithm.upper_bound - mean)/sigma_converted)));
		
			 	return mean + sigma_converted * static_cast<DType>(probit(unif_distr(generator)));
			 }

			 // Cumulative distribution function of a standard normal (phi function)
			 Real normal_cdf(const Real& x) const;

			 // Utility to compute the probit function, i.e. the quantile function of a standard normal
			 Real probit(const Real& u) const;

			 // Initialization step
			 void initialization(void);

			 // Evaluation step
			 VectorXctype evaluation(void) const;

			 // Selection and Variation steps
			 void selection_and_variation(VectorXctype values, Real alpha);
			 

	public: // Constructors
			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, 
			const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_,
			const unsigned int& max_iterations_genetic_algorithm_)
			: F(F_), best(init), param_genetic_algorithm(param_genetic_algorithm_), 
			max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_)
			 {
			 	population.resize(param_genetic_algorithm.N);
				population[0] = init;
				min_value = F(init);

				seed = std::chrono::system_clock::now().time_since_epoch().count();
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 50u) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


template <class DType>
struct Parameter_Gradient_Descent_fd
{
	DType lower_bound;	// Lower bound for input values
	DType upper_bound;	// Upper bound for output values
	DType periods;		// Periodicity for each input component (if periods[i] = 0.0 then no periodicity for i-th component)	
};


// DType = Domain variable type, CType = Codomain variable type;
// DType can have one or more components, instead CType can be only a scalar type (1 component only)
template <class DType, class CType>
class Gradient_Descent_fd
{
	private: // Function to optimize
			 std::function<CType (DType)> F;

			 // Approx of the gradient
			 std::function<DType (DType)> dF;

			 // Minimizer of F and minimal value of F
			 DType best;

			 // Gradient Descent parameters
			 Parameter_Gradient_Descent_fd<DType> param_gradient_descent_fd;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if  increment < tolerance 
			 bool goOn = true;

			 const unsigned int max_iterations_gradient_descent_fd;
			 const Real tol_gradient_descent_fd;

			 // DType vectorial case
			 template <class SFINAE = DType>
			 const typename std::enable_if< !std::is_floating_point<SFINAE>::value, Real>::type
			 compute_increment(const DType& new_sol, const DType& old_sol)
			 {	
			 	DType diff_sol(new_sol.size());
				std::transform(new_sol.data(), new_sol.data() + new_sol.size(), old_sol.data(), diff_sol.data(), std::minus<Real>()); // Compute (new_sol - old_sol) in diff_sol

			 	Real res = std::inner_product(diff_sol.data(), diff_sol.data() + diff_sol.size(), diff_sol.data(), 0.0);
		
			 	return std::sqrt(res);
			 }

			 // DType = Scalar case
			 template <class SFINAE = DType>
			 const typename std::enable_if< std::is_floating_point<SFINAE>::value, Real>::type
			 compute_increment(const DType& new_sol, const DType& old_sol)
			 {

			 	return std::sqrt((new_sol - old_sol) * (new_sol - old_sol));
			 }

			 // DType vectorial case
			 // Function to upgrade the solution in the vectorial case
			 template <class SFINAE = DType>
			 typename std::enable_if< !std::is_floating_point<SFINAE>::value, void>::type
			 upgrade_best()
			 {	
			 	DType new_best = best;

			 	// Set some useful parameters
			 	Real alpha = 1.0;
			 	Real beta = 0.5;
				CType f = F(best);
				DType df = dF(best);
				Real df_norm = std::inner_product(df.data(), df.data() + df.size(), df.data(), 0.0);

				// Update best
				std::transform(best.data(),best.data() + best.size(),df.data(),new_best.data(),[alpha](Real x,Real y){return x - alpha * y;});
			 	
			 	// Check bounds
			 	for(unsigned int i = 0u; i < best.size(); ++i)
			 	{
			 		// Exploit periodicity if present
			 		if(param_gradient_descent_fd.periods[i] != 0.0)
			 		{
						while(new_best[i] < param_gradient_descent_fd.lower_bound[i])
							new_best[i] += param_gradient_descent_fd.periods[i];
						while(new_best[i] > param_gradient_descent_fd.upper_bound[i])
							new_best[i] -= param_gradient_descent_fd.periods[i];
			 		}

			 		else
			 		{	
			 			// Check upper and lower bounds
			 			while(new_best[i] < param_gradient_descent_fd.lower_bound[i] || new_best[i] > param_gradient_descent_fd.upper_bound[i])
			 			{	
			 				new_best[i] = best[i];			 				
			 				alpha *= beta; // Re-upgrade best with a smaller alpha in order to remain inside the bounds
			 				new_best[i] -= alpha*df[i];
			 			}

			 		}
			 	}

			 	// Backtracking line search to set the step size
			 	while(F(new_best) > f - alpha * df_norm / 2.0){
			 		alpha *= beta;
			 		std::transform(best.data(),best.data() + best.size(),df.data(),new_best.data(),[alpha](Real x,Real y){return x - alpha * y;});
			 	}
				
			 	best = new_best;

			 	return;
			 }


			 // DType = Scalar case
			 // Function to upgrade the solution in the scalar case
			 template <class SFINAE = DType>
			 typename std::enable_if< std::is_floating_point<SFINAE>::value, void>::type
			 upgrade_best()
			 {

			 	// Set some useful parameters
			 	Real alpha = 1.0;
			 	Real beta = 0.5;
				CType f = F(best);
				DType df = dF(best);
				Real df_norm = df*df;
				
				// Upgrade best
			 	DType new_best = best - alpha*df;

			 	// Check bounds
			 	// Exploit periodicity if present
			 	if(param_gradient_descent_fd.periods != 0.0)
			 	{
					while(new_best < param_gradient_descent_fd.lower_bound)
						new_best += param_gradient_descent_fd.periods;
					while(new_best > param_gradient_descent_fd.upper_bound)
						new_best -= param_gradient_descent_fd.periods;
			 	}

			 	else
			 	{
			 		// Check lower and upper bounds
			 		while(new_best < param_gradient_descent_fd.lower_bound || new_best > param_gradient_descent_fd.upper_bound)
			 		{	
			 			alpha *= beta; // Re-upgrade best with a smaller alpha in order to remain inside the bounds
			 			new_best = best - alpha*df;
			 		}
			 	}
			 	
			 	// Backtracking line search to set the step size
			 	while(F(new_best) > f - alpha * df_norm / 2.0){
			 		alpha *= beta;
			 		new_best = best - alpha * df;
			 	}

			 	best = new_best;

			 	return;

			 }
			 
	public: // Constructors
			Gradient_Descent_fd(const std::function<CType (DType)>& F_, const std::function<DType (DType)>& dF_, 
			const DType& init, const Parameter_Gradient_Descent_fd<DType>& param_gradient_descent_fd_,
			const unsigned int& max_iterations_gradient_descent_fd_,
			const Real tol_gradient_descent_fd_)
			: F(F_), dF(dF_), best(init), param_gradient_descent_fd(param_gradient_descent_fd_),
			max_iterations_gradient_descent_fd(max_iterations_gradient_descent_fd_),
			tol_gradient_descent_fd(tol_gradient_descent_fd_)
			 {
			 	best = init;
			 };

			Gradient_Descent_fd(const std::function<CType (DType)>& F_, const std::function<DType (DType)>& dF_, const DType& init,
				const Parameter_Gradient_Descent_fd<DType>& param_gradient_descent_fd_)
			: Gradient_Descent_fd(F_, dF_, init, param_gradient_descent_fd_, 50u, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


template <class DType>
struct Parameter_LBFGS
{
	DType lower_bound;	// Lower bound for input values
	DType upper_bound;	// Upper bound for output values
	DType periods;		// Periodicity for each input component (if periods[i] = 0.0 then no periodicity for i-th component)	
};


// DType = Domain variable type, CType = Codomain variable type;
// DType can have one or more components, instead CType can be only a scalar type (1 component only)
template <class DType, class CType>
class LBFGS
{
	private: // Function to optimize
			 std::function<CType (DType)> F;

			 // Approx of the gradient
			 std::function<DType (DType)> dF;

			 // Minimizer of F and minimal value of F
			 DType best;

			 // Variables to store useful parameters of previous updates of LBFGS
			 std::deque<DType> s;
			 std::deque<DType> y;
			 std::deque<Real> rho;

			 // Gradient Descent parameters
			 Parameter_LBFGS<DType> param_lbfgs;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if  increment < tolerance 
			 bool goOn = true;

			 const UInt max_iterations_lbfgs;
			 const Real tol_lbfgs;

			 // DType vectorial case
			 template <class SFINAE = DType>
			 const typename std::enable_if< !std::is_floating_point<SFINAE>::value, Real>::type
			 compute_increment(const DType& new_sol, const DType& old_sol)
			 {	
			 	DType diff_sol(new_sol.size());
				std::transform(new_sol.data(), new_sol.data() + new_sol.size(), old_sol.data(), diff_sol.data(), std::minus<Real>()); // Compute (new_sol - old_sol) in diff_sol

			 	return std::sqrt(dot_prod(diff_sol, diff_sol));
			 }

			 // DType = Scalar case
			 template <class SFINAE = DType>
			 const typename std::enable_if< std::is_floating_point<SFINAE>::value, Real>::type
			 compute_increment(const DType& new_sol, const DType& old_sol)
			 {

			 	return std::sqrt((new_sol - old_sol) * (new_sol - old_sol));
			 }

			 // DType vectorial case
			 template <class SFINAE = DType>
			 const typename std::enable_if< !std::is_floating_point<SFINAE>::value, Real>::type
			 dot_prod(const DType& v1, const DType& v2)
			 {	
			 	return std::inner_product(v1.data(), v1.data() + v1.size(), v2.data(), 0.0);
			 }

			 // DType vectorial case
			 // Function to upgrade the solution in the vectorial case
			 template <class SFINAE = DType>
			 typename std::enable_if< !std::is_floating_point<SFINAE>::value, void>::type
			 upgrade_best(UInt iter)
			 {	
			 	DType new_best = best;

			 	// Set some useful parameters
			 	Real alpha = 1.0;
			 	Real beta = 0.5;
				CType f = F(best);
				DType df = dF(best);
				DType q = df;

				UInt m = s.size();

				std::vector<Real> a(m);

				for(UInt i = 0u; i < m; ++i)
				{
					a[i] = dot_prod(s[i], q) * rho[i];
					q -= a[i] * y[i]; 
				}

				Real gamma = (m == 0) ? 1.0 : (dot_prod(s[0], y[0])) / dot_prod(y[0],y[0]);
				
				Eigen::MatrixXd H = Eigen::MatrixXd::Identity(best.size(), best.size()) * gamma;
				
				q = H*q; // KO con std::vector

				for(UInt i = m - 1; i >= 0; --i)
				{	
					q += (a[i] - rho[i] * dot_prod(y[i], q)) * s[i]; // KO con std::vector
				}

				// Update best
				std::transform(best.data(),best.data() + best.size(),q.data(),new_best.data(),[alpha](Real x,Real y){return x - alpha * y;});
			 	
			 	// Check bounds
			 	for(unsigned int i = 0u; i < best.size(); ++i)
			 	{
			 		// Exploit periodicity if present
			 		if(param_lbfgs.periods[i] != 0.0)
			 		{
						while(new_best[i] < param_lbfgs.lower_bound[i])
							new_best[i] += param_lbfgs.periods[i];
						while(new_best[i] > param_lbfgs.upper_bound[i])
							new_best[i] -= param_lbfgs.periods[i];
			 		}

			 		else
			 		{	
			 			// Check upper and lower bounds
			 			while(new_best[i] < param_lbfgs.lower_bound[i] || new_best[i] > param_lbfgs.upper_bound[i])
			 			{	
			 				new_best[i] = best[i];			 				
			 				alpha *= beta; // Re-upgrade best with a smaller alpha in order to remain inside the bounds
			 				new_best[i] -= alpha*q[i];
			 			}

			 		}
			 	}

				s.push_front(new_best - best); // KO con std::vector (differences not allowed)
				y.push_front(dF(new_best) - df); // KO con std::vector
				rho.push_front(1.0 / dot_prod(y[0], s[0]));

			 	best = new_best;

			 	if(m == 5)
			 	{
			 		s.pop_back();
			 		y.pop_back();
			 		rho.pop_back();
			 	}

			 	return;
			 }

			 // DType scalar case
			 // Function to upgrade the solution in the vectorial case
			 template <class SFINAE = DType>
			 typename std::enable_if< std::is_floating_point<SFINAE>::value, void>::type
			 upgrade_best(UInt iter)
			 {	
			 	// Set some useful parameters
			 	Real alpha = 1.0;
			 	Real beta = 0.5;
				DType df = dF(best);
				DType q = df;

				UInt m = s.size();

				Real gamma = (m == 0) ? 1.0 : s[0] / y[0];
				
				q *= gamma;

				// Update best
				DType new_best = best - alpha * q;
			 	
			 	// Check bounds
			 	// Exploit periodicity if present
			 	if(param_lbfgs.periods != 0.0)
			 	{
					while(new_best < param_lbfgs.lower_bound)
						new_best += param_lbfgs.periods;
					while(new_best > param_lbfgs.upper_bound)
						new_best -= param_lbfgs.periods;
			 	}

			 	else
			 	{
			 		// Check lower and upper bounds
			 		while(new_best < param_lbfgs.lower_bound || new_best > param_lbfgs.upper_bound)
			 		{	
			 			alpha *= beta; // Re-upgrade best with a smaller alpha in order to remain inside the bounds
			 			new_best = best - alpha * q;
			 		}
			 	}
			 	
			 	s.push_front(new_best - best); // KO con std::vector (differences not allowed)
				y.push_front(dF(new_best) - df); // KO con std::vector

			 	best = new_best;

			 	if(m != 0)
			 	{
			 		s.pop_back();
				 	y.pop_back();
			 	}

			 	return;
			 }

	public: // Constructors
			LBFGS(const std::function<CType (DType)>& F_, const std::function<DType (DType)>& dF_, 
			const DType& init, const Parameter_LBFGS<DType>& param_lbfgs_,
			const UInt& max_iterations_lbfgs_,
			const Real tol_lbfgs_)
			: F(F_), dF(dF_), best(init), param_lbfgs(param_lbfgs_),
			max_iterations_lbfgs(max_iterations_lbfgs_),
			tol_lbfgs(tol_lbfgs_)
			 {
			 	best = init;
			 };

			LBFGS(const std::function<CType (DType)>& F_, const std::function<DType (DType)>& dF_, const DType& init,
				const Parameter_LBFGS<DType>& param_lbfgs_)
			: LBFGS(F_, dF_, init, param_lbfgs_, 50, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};



#include "Optimization_Algorithm_imp.h"

#endif
