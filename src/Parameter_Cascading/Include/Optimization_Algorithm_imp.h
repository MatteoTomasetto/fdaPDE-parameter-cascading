#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__

#include <array>
#include <algorithm>
#include <type_traits>

// Cumulative distribution function of a standard normal (phi function)
template <class DType, class CType>
Real Genetic_Algorithm<DType, CType>::normal_cdf(const Real& x) const
{
	return 0.5 + 0.5 * std::erf(x * M_SQRT1_2);
}

// Inverse cumulative distribution function of a standard normal (probit function)
// approximated via Beasley-Springer-Moro algorithm.
template <class DType, class CType>
Real Genetic_Algorithm<DType, CType>::probit(const Real& u) const
{
	std::array<Real, 4> a = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};

	std::array<Real, 4> b = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};

	std::array<Real, 9> c = {0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
							 0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
							 0.0000003960315187};

	Real y = u - 0.5;

	if(std::abs(y) < 0.42)
	{
		Real r = y * y;
		Real num = a[3];
		Real den = b[3];

		for (UInt i = 2; i >= 0; --i)
		{
			num = num*r + a[i];
			den = den*r + b[i];
		}

		den = den*r + 1.0;
		
		return y * num / den; 
	}
	else
	{
		Real r = (y > 0.0) ? 1.0 - u : u;

		r = std::log(-std::log(r));

		Real x = c[8];
		
		for (UInt i = 7; i >= 0; --i)
		{
			x = x * r + c[i];
		}

		if (y < 0)
			x *= -1;

		return x;

	}

}

template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::initialization(void)
{	
	Real sigma = 2.0; // Not small standard deviation to explore enough a region near population[0]

	// Populate the candidate solutions in population
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i){
		population[i] = get_random_element(population[0], sigma);
	}

	return;
}


template <class DType, class CType>
typename Genetic_Algorithm<DType, CType>::VectorXctype Genetic_Algorithm<DType, CType>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res[i] = F(population[i]); // Evaluate F in each candidate solution in population

	return res;
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one in 3 different ways:
	// 		a) winner + little gaussian noise
	//		b) winner + larger gaussian noise
	// 		c) best + little gaussian noise
	std::default_random_engine generator(seed++);
	std::uniform_int_distribution<UInt> dice(0, 3);

	Real sigma1 = 0.5;
	Real sigma2 = 1.5;

	for (unsigned int i = 0u; i < param_genetic_algorithm.N - 1; i += 2u)
	{	
		unsigned int idx_loser = values[i] > values[i+1] ? i : i + 1;
		unsigned int idx_winner = idx_loser == i ? i + 1 : i;
		UInt choice = dice(generator);
		
		if(choice == 0)
			population[idx_loser] = get_random_element(population[idx_winner], sigma1);
		else if(choice == 1)
			population[idx_loser] = get_random_element(population[idx_winner], sigma2);
		else
			population[idx_loser] = get_random_element(best, sigma1);
	}

	return;
}

template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::apply(void)
{	
	initialization();
	
	// Evaluate the loss function for population elements: this is needed for selection process
	VectorXctype F_values(param_genetic_algorithm.N);
	F_values = evaluation();
	
	unsigned int iter = 0u;
	unsigned int counter = 0u; // counter how many times "best" does not change

	while(iter < max_iterations_genetic_algorithm && goOn)
	{	
		++iter;

		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values);
		
		// Find the best solution of this iteration
		F_values = evaluation();
		auto ptr_min_value = std::min_element(F_values.begin(), F_values.end());
    	UInt best_index = std::distance(F_values.begin(), ptr_min_value);

		// If the new solution beats the current best solution
		if(min_value > *ptr_min_value)
		{
			best = population[best_index];
			min_value = *ptr_min_value;
			counter = 0u;
		}
		else
			++counter;
		
		goOn = counter < 3;
		
}
	
	return;
}

#endif