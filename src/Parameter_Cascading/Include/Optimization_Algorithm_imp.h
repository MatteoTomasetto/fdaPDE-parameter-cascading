#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__

#include <random>
#include <chrono>
#include <algorithm>
#include <type_traits>


// Random element generator used if DType is a container (vector with more than one elements)
template <class DType, class CType, class SFINAE>
const DType Genetic_Algorithm<DType, CType, SFINAE>::get_random_element(const DType& mean, const Real& sigma) const
{	
	std::default_random_engine generator{std::random_device{}()};

	unsigned int ElemSize = mean.size();

	DType res;
	res.resize(ElemSize);
	
	for(unsigned int j = 0u; j < ElemSize; ++j) // loop over each component of mean and res
	{	
		// Set lower and upper values for the uniform distribution
		Real a = std::max(mean(j) - sigma, param_genetic_algorithm.lower_bound(j));
		Real b = std::min(mean(j) + sigma, param_genetic_algorithm.upper_bound(j));

		// Generate the j-th random component
		std::uniform_real_distribution<double> distr(a, b); // not general, we should use decltype(mean(j)) instead of double

		res(j) = distr(generator);
	}

	return res;
}


// Random element generator used if DType is a scalar
template <class DType, class CType>
const DType Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::get_random_element(const DType& mean, const Real& sigma) const
{	
	std::default_random_engine generator{time(0)};

	// Set lower and upper values for the uniform distribution
	Real a = std::max(mean - sigma, param_genetic_algorithm.lower_bound);
	Real b = std::min(mean + sigma, param_genetic_algorithm.upper_bound);
	std::uniform_real_distribution<DType> distr(a, b);

	return distr(generator);
}


template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::initialization(void)
{	
	const Real sigma = 1.5;

	// Populate the candidate solutions container "population"
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population[i] = get_random_element(population[0], sigma);

	return;
}

template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::initialization(void)
{	
	const Real sigma = 1.5;

	// populate the candidate solutions container "population"
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population[i] = get_random_element(population[0], sigma);

	return;
}


template <class DType, class CType, class SFINAE>
typename Genetic_Algorithm<DType, CType, SFINAE>::VectorXctype Genetic_Algorithm<DType, CType, SFINAE>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res(i) = F(population[i]);

	return res;
}

template <class DType, class CType>
typename Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::VectorXctype Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res(i) = F(population[i]);

	return res;
}

// selection of best candidates and variation to investigate new solutions (case when DType is a vector)
template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 
	const Real sigma = 1.0;
	DType zero_elem(population[0].size());

	std::fill(zero_elem.data(), zero_elem.data() + zero_elem.size(), 0.0);

	unsigned int i = 0u;

	for (; i < param_genetic_algorithm.N - 2; i += 2u)
	{	
		if(values(i) > values(i+1))
		{
			population[i] = population[i+1] + get_random_element(zero_elem, sigma);
		}

		else
			population[i+1] = population[i] + get_random_element(zero_elem, sigma);
	}

	if(i == param_genetic_algorithm.N - 2) // Extra case to manage the case with N odd
	{
		if(values(i) > values(i+1))
			population[i] = population[i+1] + get_random_element(zero_elem, sigma);
		else
			population[i+1] = population[i] + get_random_element(zero_elem, sigma);

	}

	return;
}


// selection of best candidates and variation to investigate new solutions (case when DType is a scalar)
template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value, void >::type>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 
	const Real sigma = 1.0;
	DType zero_elem;

	zero_elem = static_cast<DType>(0);
	
	unsigned int i = 0u;

	for (; i < param_genetic_algorithm.N - 2; i += 2u)
	{	
		if(values(i) > values(i+1))
		{
			population[i] = population[i+1] + get_random_element(zero_elem, sigma);
		}

		else
			population[i+1] = population[i] + get_random_element(zero_elem, sigma);
	}

	if(i == param_genetic_algorithm.N - 2) // Extra case to manage the case with N odd
	{
		if(values(i) > values(i+1))
			population[i] = population[i+1] + get_random_element(zero_elem, sigma);
		else
			population[i+1] = population[i] + get_random_element(zero_elem, sigma);

	}

	return;
}


// mutation step when DType is a vector
template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::mutation(void)
{	
	std::default_random_engine generator{time(0)};

	std::uniform_real_distribution<double> distr(0.0, 1.0);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
	{	
		if(distr(generator) < 0.25)  // mutation for 1/4 of the population on average
			std::swap(population[i](0), population[i](1));
	}
	return;
}


// mutation step when DType is a scalar
template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::mutation(void)
{	
	return;
}

// Compute increment between old and new solution (case when DType is a vector)
template <class DType, class CType, class SFINAE>
Real Genetic_Algorithm<DType, CType, SFINAE>::compute_increment(DType new_sol, DType old_sol) const
{
	return (new_sol - old_sol).norm();
}


// Compute increment between old and new solution (case when DType is a scalar)
template <class DType, class CType>
Real Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::compute_increment(DType new_sol, DType old_sol) const
{
	return std::abs(new_sol - old_sol);
}


template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::apply(void)
{	
	initialization();
	Rprintf("initialization done\n");

	// Evaluate the loss function for population elements: this is needed for selection process
	VectorXctype F_values(param_genetic_algorithm.N);
	F_values = evaluation();
	Rprintf("evaluation done\n");

	unsigned int iter = 0u;

	while(iter < max_iterations_genetic_algorithm && goOn)
	{	
		// Save the old solution to compute error
		DType old_solution = best;

		++iter;

		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values);
		Rprintf("selection and variation done\n");
		mutation();
		Rprintf("mutation done\n");

		// Find the best solution
		F_values = evaluation();
		Rprintf("evaluation done\n");
		UInt best_index;
		F_values.minCoeff(&best_index);
		best = population[best_index];

		double best1 = best(0);
		double best2 = best(1);
		Rprintf("best angle and intensity %f , %f \n", best1, best2);

		goOn = compute_increment(best, old_solution) > tol_genetic_algorithm;

	}
	
	return;
}

template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::apply(void)
{	
	initialization();
	Rprintf("initialization done\n");

	// Evaluate the loss function for population elements: this is needed for selection process
	VectorXctype F_values(param_genetic_algorithm.N);
	F_values = evaluation();
	Rprintf("evaluation done\n");


	unsigned int iter = 0u;

	while(iter < max_iterations_genetic_algorithm && goOn)
	{	
		// Save the old solution to compute error
		DType old_solution = best;

		++iter;

		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values);
		Rprintf("selection and variation done\n");
		mutation();
		Rprintf("mutation done\n");

		// Find the best solution
		F_values = evaluation();
		Rprintf("evaluation done\n");
		UInt best_index;
		F_values.minCoeff(&best_index);
		best = population[best_index];

		Rprintf("best %f\n", best);

		goOn = compute_increment(best, old_solution) > tol_genetic_algorithm;

	}
	
	return;
}

#endif