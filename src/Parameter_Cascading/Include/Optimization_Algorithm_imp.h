#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__

#include <array>
#include <random>
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
		Real a = std::max(mean[j] - sigma, param_genetic_algorithm.lower_bound[j]);
		Real b = std::min(mean[j] + sigma, param_genetic_algorithm.upper_bound[j]);

		// Generate the j-th random component from a normal truncated in (a,b)
		std::uniform_real_distribution<decltype(mean[0])> unif_distr(normal_cdf((a - mean[j])/sigma), normal_cdf((b - mean[j])/sigma));

		res[j] = mean[j] + sigma*probit(unif_distr(generator));
	}

	return res;
}

// Random element generator used if DType is a scalar
template <class DType, class CType>
const DType Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::get_random_element(const DType& mean, const Real& sigma) const
{	
	std::default_random_engine generator{std::random_device{}()};

	// Set lower and upper values for the uniform distribution
	Real a = std::max(mean - sigma, param_genetic_algorithm.lower_bound);
	Real b = std::min(mean + sigma, param_genetic_algorithm.upper_bound);

	// Generate the a number from a normal truncated in (a,b)
	std::uniform_real_distribution<DType> unif_distr(normal_cdf((a - mean)/sigma), normal_cdf((b - mean)/sigma));

	return mean + sigma*probit(unif_distr(generator));
}


// Cumulative distribution function of a standard normal (phi function)
template <class DType, class CType, class SFINAE>
Real Genetic_Algorithm<DType, CType, SFINAE>::normal_cdf(const Real& x) const {
	return 0.5 + 0.5 * std::erf(x * M_SQRT1_2);
}

// Inverse cumulative distribution function of a standard normal (probit function)
// approximated via Beasley-Springer-Moro algorithm.
template <class DType, class CType, class SFINAE>
Real Genetic_Algorithm<DType, CType, SFINAE>::probit(const Real& u) const {
	
	std::array<Real, 4> a = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};

	std::array<Real, 4> b = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};

	std::array<Real, 9> c = {0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
							 0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
							 0.0000003960315187};

	if(u >= 0.5 && u <= 0.92)
	{
		Real num = 0.0;
    	Real denom = 1.0;

		for (unsigned int i = 0u; i < 4u; ++i)
		{
			num += a[i] * std::pow((u - 0.5), 2u * i + 1u);
			denom += b[i] * std::pow((u - 0.5), 2u * i);
		}
    
		return num/denom;
	
	}

	else if (u > 0.92 && u < 1)
	{
		Real num = 0.0;

		for (unsigned int i = 0u; i < 9u; ++i)
		{
			num += c[i] * std::pow((std::log(-std::log(1 - u))), i);
		}

		return num;
	}

	else
	{
		return -1.0*inv_cdf(1-u);
	}
}


template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::initialization(void)
{	
	const Real sigma = 2.0;

	// Populate the candidate solutions in population
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population[i] = get_random_element(population[0], sigma);

	return;
}

template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::initialization(void)
{	
	const Real sigma = 2.0;

	// Populate the candidate solutions container "population"
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population[i] = get_random_element(population[0], sigma);

	return;
}


template <class DType, class CType, class SFINAE>
typename Genetic_Algorithm<DType, CType, SFINAE>::VectorXctype Genetic_Algorithm<DType, CType, SFINAE>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res[i] = F(population[i]);

	return res;
}

template <class DType, class CType>
typename Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::VectorXctype Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res[i] = F(population[i]);

	return res;
}

// Selection of best candidates and variation to investigate new solutions (case when DType is a vector)
template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 
	const Real sigma = 0.5;

	unsigned int i = 0u;

	for (; i < param_genetic_algorithm.N - 2; i += 2u)
	{	
		if(values[i] > values[i+1]) // ATTENTION: if CType è vettore?... potremmo confrontare norme...
		{
			population[i] = get_random_element(population[i+1], sigma);
		}

		else
			population[i+1] = get_random_element(population[i+1], sigma);
	}

	if(i == param_genetic_algorithm.N - 2) // Extra case to manage the case with N odd
	{
		if(values[i] > values[i+1])
			population[i] = get_random_element(population[i+1], sigma);
		else
			population[i+1] = get_random_element(population[i], sigma);

	}

	return;
}


// Selection of best candidates and variation to investigate new solutions (case when DType is a scalar)
template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value, void >::type>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 
	const Real sigma = 0.5;
	
	unsigned int i = 0u;

	for (; i < param_genetic_algorithm.N - 2; i += 2u)
	{	
		if(values[i] > values[i+1])
		{
			population[i] = get_random_element(population[i+1], sigma);
		}

		else
			population[i+1] = get_random_element(population[i], sigma);
	}

	if(i == param_genetic_algorithm.N - 2) // Extra case to manage the case with N odd
	{
		if(values[i] > values[i+1])
			population[i] = get_random_element(population[i+1], sigma);
		else
			population[i+1] = get_random_element(population[i], sigma);

	}

	return;
}


// Mutation step when DType is a vector
template <class DType, class CType, class SFINAE>
void Genetic_Algorithm<DType, CType, SFINAE>::mutation(void)
{	
	std::default_random_engine generator{std::random_device{}()};

	std::uniform_real_distribution<Real> distr(0.0, 1.0);
	std::uniform_int_distribution<UInt> distr_indices(0, param_genetic_algorithm.N - 1);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
	{	
		if(distr(generator) < 0.25)  // mutation (crossover) for 1/4 of the population on average
		{
			idx1 = distr_indices(generator);
			idx2 = distr_indices(generator);
			std::swap(population[idx1][0], population[idx2][0]);
		}
	}
	return;
}


// Mutation step when DType is a scalar
template <class DType, class CType>
void Genetic_Algorithm<DType, CType, typename std::enable_if< std::is_floating_point<DType>::value >::type>::mutation(void)
{	
	return;
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
	unsigned int counter = 0u; // counter how many times "best" does not change

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

		// Find the best solution of this iteration
		F_values = evaluation();
		Rprintf("evaluation done\n");
		UInt best_index;
		F_values.minCoeff(&best_index);
		
		// New solution beats the current best solution
		if(min_value > F_values[best_index])
		{
			best = population[best_index];
			counter = 0u;
		}

		else
			++counter;
		
		goOn = counter < 5;



		// CHECK
		double best1 = best(0);
		double best2 = best(1);
		Rprintf("best angle and intensity %f , %f \n", best1, best2);


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
	unsigned int counter = 0u; // counter how many times "best" does not change

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
		
		Rprintf("best %f\n", best);

		// New solution beats the current best solution
		if(min_value > F_values[best_index])
		{
			best = population[best_index];
			counter = 0u;
		}

		else
			++counter;
		
		goOn = counter < 5;

	}
	
	return;
}

#endif