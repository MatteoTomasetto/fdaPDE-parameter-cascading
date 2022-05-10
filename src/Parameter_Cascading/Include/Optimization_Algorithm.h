#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"
#include <functional>

struct Parameter_Genetic_Algorithm
{
	unsigned int N; 	// population size
	Real prob_mutation; // probability of mutation
	Real prob_crossover;// probability of crossover
}

template <class DType, class CType> // DType=Domain, CType = Codomain;
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

			 // Initialization step
			 void initialization(void);

			 // Evaluation step
			 void evaluation(void);

			 // Selection step
			 void selection(VectorXctype values);

			 // Variation step
			 void variation(void);

			 // Replacement step
			 void replacement(void);			 			 

	public: // Constructors
			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, 
			const Parameter_Genetic_Algorithm& param_genetic_algorithm_, const unsigned int& max_iterations_genetic_algorithm_,
			const Real & tol_genetic_algorithm_)
			: F(F_), max_iterations_genetic_algorithm(max_iterations_genetic_algorithm_),
			tol_genetic_algorithm(tol_genetic_algorithm_)
			 {
			 	population << init;
			 	best = init;
			 };

			Genetic_Algorithm(const std::function<CType (DType)>& F_, const DType& init, const Parameter_Genetic_Algorithm& param_genetic_algorithm_)
			: Genetic_Algorithm(F_, init, param_genetic_algorithm_, 100u, 1e-3) {};

			// Function to apply the algorithm
			void apply(void);

			// Getters
			inline DType get_solution(void) {return best;};
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