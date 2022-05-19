#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__

#include <random>
#include <algorithm>
#include <type_traits>


// Random element generator used if DType is a container (vector with more than one elements)
template <class DType, class CType>
const DType& Genetic_Algorithm<DType, CType>::get_random_element(const DType& mean, const Real& sigma) const
{	
	std::random_device rd;
	std::knuth_b seed{rd()};
	std::default_random_engine generator{seed};

	if(std::is_scalar<DType>::value)
	{
		// Set lower and upper values for the uniform distribution
		Real a = std::max(mean - sigma, param_genetic_algorithm.lower_bound);
		Real b = std::min(mean + sigma, param_genetic_algorithm.upper_bound);
		std::uniform_real_distribution<DType> distr(a, b);

		return distr(generator);
	}

	else
	{
		unsigned int ElemSize = mean.size();

		DType res;
		res.resize(ElemSize);
	
		for(unsigned int j = 0u; j < ElemSize; ++j) // loop over each component of mean and res
		{	
			// Set lower and upper values for the uniform distribution
			Real a = std::max(mean(j) - sigma, param_genetic_algorithm.lower_bound(j));
			Real b = std::min(mean(j) + sigma, param_genetic_algorithm.upper_bound(j));

			// Generate the j-th random component
			std::uniform_real_distribution<decltype(mean(j))> distr(a, b);

			res(j) = distr(generator);
		}

		return res;
	}

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::initialization(void)
{	
	const Real sigma = 1.5;

	// populate the candidate solutions container "population"
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population(i) = get_random_element(population(0), sigma);

	return;
}


template <class DType, class CType>
typename Genetic_Algorithm<DType, CType>::VectorXctype Genetic_Algorithm<DType, CType>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res(i) = F(population(i));

	return res;
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 
	const Real sigma = 1.0;
	const DType zero_elem(population(0).size());

	if constexpr (std::is_scalar<DType>::value)
		zero_elem = static_cast<DType>(0);
	else
		std::fill(zero_elem.data(), zero_elem.data() + zero_elem.size(), 0.0);

	unsigned int i = 0u;

	for (; i < param_genetic_algorithm.N - 2; i += 2u)
	{	
		if(values(i) > values(i+1))
		{
			population(i) = population(i+1) + get_random_element(zero_elem, sigma);
		}

		else
			population(i+1) = population(i) + get_random_element(zero_elem, sigma);
	}

	if(i == param_genetic_algorithm.N - 2) // Extra case to manage the case with N odd
	{
		if(values(i) > values(i+1))
			population(i) = population(i+1) + get_random_element(zero_elem, sigma);
		else
			population(i+1) = population(i) + get_random_element(zero_elem, sigma);

	}

	return;
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::mutation(void)
{	
	if constexpr (std::is_scalar<DType>::value)
		return;
	else
	{
		std::random_device rd;
		std::knuth_b seed{rd()};
		std::default_random_engine generator{seed};
		std::uniform_int_distribution<> distr(0, param_genetic_algorithm.N);
		
		for(unsigned int i = 0u; i < param_genetic_algorithm.N / 4; ++i) // mutation for 1/4 of the population
		{
			UInt idx1 = distr(generator);
			UInt idx2 = distr(generator);

			std::swap(population(idx1)(0), population(idx2)(1));
		}
	}	

	return;
}


// Compute increment between old and new solution
template <class DType, class CType>
Real Genetic_Algorithm<DType, CType>::compute_increment(DType new_sol, DType old_sol) const
{
	if constexpr (std::is_scalar<DType>::value)
		return std::abs(new_sol - old_sol);
	else
		return (new_sol - old_sol).norm();
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::apply(void)
{	
	initialization();

	// Evaluate the loss function for population elements: this is needed for selection process
	VectorXctype F_values(param_genetic_algorithm.N);
	F_values = evaluation();

	unsigned int iter = 0u;

	while(iter < max_iterations_genetic_algorithm && goOn)
	{	
		// Save the old solution to compute error
		DType old_solution = best;

		++iter;

		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values);
		mutation();

		// Find the best solution
		F_values = evaluation();
		UInt best_index;
		F_values.minCoeff(&best_index);
		best = population(best_index);

		goOn = compute_increment(best, old_solution) > tol_genetic_algorithm;

	}
	
	return;
}



#endif