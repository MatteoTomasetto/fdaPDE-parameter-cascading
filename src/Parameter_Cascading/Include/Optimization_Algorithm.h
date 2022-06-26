#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include <functional>
#include <vector>
#include <type_traits>
#include <random>
#include <chrono>

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
			 get_random_element(const DType& mean, Real& sigma)
			 {
			 	std::default_random_engine generator(seed++);
			 	
			 	unsigned int ElemSize = mean.size();
			 	DType res;
			 	res.resize(ElemSize);

			 	sigma = static_cast<typename DType::value_type>(sigma);

			 	// Loop over each component of mean and res and generate random the components of res
			 	for(unsigned int j = 0u; j < ElemSize; ++j)
			 	{	
			 		// Generate the j-th random component from a normal truncated wrt lower and upper bound in param_genetic_algorithm
			 		std::uniform_real_distribution<Real>
			 		unif_distr(normal_cdf(static_cast<Real>((param_genetic_algorithm.lower_bound[j] - mean[j])/sigma)),
			 				   normal_cdf(static_cast<Real>((param_genetic_algorithm.upper_bound[j] - mean[j])/sigma)));

			 		res[j] = mean[j] + sigma * static_cast<typename DType::value_type>(probit(unif_distr(generator)));
			 	}
	
			 	return res;
			 }
			 // DType = Scalar case
			 template <class SFINAE = DType>
			 const typename std::enable_if< std::is_floating_point<SFINAE>::value, DType>::type
			 get_random_element(const DType& mean, Real& sigma)
			 {

			 	sigma = static_cast<DType>(sigma);
		 	
			 	// Generate the a number from a normal truncated in (a,b)
			 	std::default_random_engine generator(seed++);
			 	std::uniform_real_distribution<Real>
			 	unif_distr(normal_cdf(static_cast<Real>((param_genetic_algorithm.lower_bound - mean)/sigma)),
			 			   normal_cdf(static_cast<Real>((param_genetic_algorithm.upper_bound - mean)/sigma)));
		
			 	return mean + sigma * static_cast<DType>(probit(unif_distr(generator)));
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
			 void selection_and_variation(VectorXctype values);
			 

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

				Rprintf("seed = %d\n", seed);
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 50u) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


#include "Optimization_Algorithm_imp.h"

#endif
