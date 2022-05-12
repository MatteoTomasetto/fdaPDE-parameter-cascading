#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include <functional>

struct Parameter_Genetic_Algorithm
{
	unsigned int N;			// population size
	DType lower_bound;		// lower bound for input values
	DType upper_bound;		// upper bound for output values
}

template <class DType, class CType> // DType=Domain variable type, CType = Codomain variable type;
class Genetic_Algorithm
{
	private: typedef Eigen::Matrix<DType,Eigen::Dynamic,1> VectorXdtype;
			 typedef Eigen::Matrix<CType,Eigen::Dynamic,1> VectorXctype;

			 // Function to optimize
			 std::function<CType (DType)> F; 

			 // Population of candidate solutions
			 VectorXdtype population;
			 
			 // Best solution (minimum point)
			 DType best;

			 // Genetic algortihm parameters
			 Parameter_Genetic_Algorithm param_genetic_algorithm;

			 // Boolean to keep looping with genetic algorithm;
			 // It becomes "false" if	|best_{k} - best_{k-1}| < tol_genetic_algorithm
			 bool goOn = true;

			 const unsigned int max_iterations_genetic_algorithm;
			 const Real tol_genetic_algorithm;

			 // Generate random DType
			 template <class SFINAE = void>
			 const DType& get_random_element(const DType& mean, const Real& sigma) const;

			 // Initialization step
			 void initialization(void);

			 // Evaluation step
			 void evaluation(void) const;

			 // Selection and Variation steps
			 void selection_and_variation(VectorXctype values);

			 // Mutation step
			 void mutation(void);	

			 // Compute error
			 Real compute_increment(DType new_sol, DType old_sol) const;		 			 

	public: // Constructors
			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, 
			const Parameter_Genetic_Algorithm& param_genetic_algorithm_, const unsigned int& max_iterations_genetic_algorithm_,
			const Real & tol_genetic_algorithm_)
			: F(F_), param_genetic_algorithm(param_genetic_algorithm_), 
			max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_),
			tol_genetic_algorithm(tol_genetic_algorithm_)
			 {
			 	population << init;
			 	population.resize(param_genetic_algorithm.N);
			 	best = init;
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 100u, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) const {return best;};
};


#include "Optimization_Algorithm_imp.h"

#endif

/* NEWTON
H.set_lambda(lambdas(iter))
Function_Wrapper<lambda::type<2>, Real, lambda::type<2>, MatrixXr, PDE_Parameter_Functional<InputCarrier>> F(H);
// TODO in PDE_Param_Functional we need compute_f, compute_fp, compute_fs;

Newton_ex<lambda::type<2>, MatrixXr, PDE_Parameter_Functional<InputCarrier>> opt_method(F);

lambda::type<2> x0 = lambda::make_pair(angles(iter), intensities(iter));
Checker ch;
std::vector<Real> Fun_values;
std::vector<lambda::type<2>> param_values;

std::pair<Real, UInt> sol = opt_method.compute(x0, tol_parameter_cascading, max_iter_parameter_cascading, Checker & ch, Fun_values, param_values);

// NB: We can use a VectorXr instead of lambda::type<2> (?).

*/