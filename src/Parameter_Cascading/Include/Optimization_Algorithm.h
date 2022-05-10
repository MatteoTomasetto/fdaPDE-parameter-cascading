/*#ifndef __GENETIC_ALGORITHM_H__
#define __GENETIC_ALGORITHM_H__

#include "../../FdaPDE.h"
#include "PDE_Parameter_Functionals.h"

//TODO: [angles(iter + 1), intensities(iter + 1)] = OPT_ALGORITHM_K(angles(iter), intensities(iter), lambdas(iter));

template <class DType, class CType>
class Genetic_Algorithm
{
	private: std::function<DType, CType> F; // this can also be a particular case of FunctionWrapper!
			 //in our case: F(x){return PDE_Parameter_Functional<InputCarrier> H.eval_K(x, lamdas(iter))}; (or we use before 
			 // H.set_lambda).

			 std::vector<DType> population;
			 
			 DType best;

			 void initialization();

			 void evaluation();

			 void selection();

			 void variation();

			 void replacement();			 			 

	public: Genetic_Algorithm(std::function<DType, CType> F_, VectorXr init) : F(F_), population(init) {};
			//init = [angles(iter), intensities(iter)]
			
			void apply();

};





















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