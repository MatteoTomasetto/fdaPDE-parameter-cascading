#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include <functional>
#include <vector>

template <class DType>
struct Parameter_Genetic_Algorithm
{
	unsigned int N;			// population size
	DType lower_bound;		// lower bound for input values
	DType upper_bound;		// upper bound for output values
};

template <class DType, class CType, class SFINAE = void> // DType = Domain variable type, CType = Codomain variable type;
class Genetic_Algorithm
{
	private: typedef std::vector<DType> VectorXdtype;
			 typedef std::vector<CType> VectorXctype;

			 // Function to optimize
			 std::function<CType (DType)> F; 

			 // Population of candidate solutions
			 VectorXdtype population;
			 
			 // Best solution (minimizer of F)
			 DType best;

			 // Genetic algortihm parameters
			 Parameter_Genetic_Algorithm<DType> param_genetic_algorithm;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if relative_increment = |best_{k} - best_{k-1}|/|best_{k}| < tol_genetic_algorithm
			 bool goOn = true;

			 const unsigned int max_iterations_genetic_algorithm;
			 const Real tol_genetic_algorithm;

			 // Generate random DType centered in "mean" and with variability sigma
			 const DType get_random_element(const DType& mean, const Real& sigma) const;

			 // Initialization step
			 void initialization(void);

			 // Evaluation step
			 VectorXctype evaluation(void) const;

			 // Selection and Variation steps
			 void selection_and_variation(VectorXctype values);

			 // Mutation step
			 void mutation(void);				 

	public: // Constructors
			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, 
			const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_,
			const unsigned int& max_iterations_genetic_algorithm_,
			const Real & tol_genetic_algorithm_)
			: F(F_), bes(init), param_genetic_algorithm(param_genetic_algorithm_), 
			max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_),
			tol_genetic_algorithm(tol_genetic_algorithm_)
			 {
			 	population.resize(param_genetic_algorithm.N);
				population[0] = init;
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 100u, 1e-2) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


// specialization via SFINAE used when DType is a scalar (not a vector)
template <class DType, class CType> // DType = Domain variable type, CType = Codomain variable type;
class Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value, void >::type>
{
	private: typedef Eigen::Matrix<DType,Eigen::Dynamic,1> VectorXdtype;
			 typedef Eigen::Matrix<CType,Eigen::Dynamic,1> VectorXctype;

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
			 // It becomes "false" if relative_increment = |best_{k} - best_{k-1}|/|best_{k}| < tol_genetic_algorithm
			 bool goOn = true;

			 const unsigned int max_iterations_genetic_algorithm;
			 const Real tol_genetic_algorithm;

			 // Generate random DType
			 const DType get_random_element(const DType& mean, const Real& sigma) const;

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

			 // Mutation step
			 void mutation(void);	

	public: // Constructors
			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, 
			const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_,
			const unsigned int& max_iterations_genetic_algorithm_,
			const Real & tol_genetic_algorithm_)
			: F(F_), best(init), param_genetic_algorithm(param_genetic_algorithm_), 
			max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_),
			tol_genetic_algorithm(tol_genetic_algorithm_)
			 {
			 	population.resize(param_genetic_algorithm.N);
				population[0] = init;
				min_value = F(init);
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm<DType>& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 100u, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


#include "Optimization_Algorithm_imp.h"

#endif
