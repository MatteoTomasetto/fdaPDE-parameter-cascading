#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__

#include <array>
#include <random>
#include <algorithm>
#include <type_traits>


template <class DType, class CType>
const DType Genetic_Algorithm<DType, CType>::get_random_element(const DType& mean, const DType& sigma) const
{	
	std::default_random_engine generator{std::random_device{}()};

	unsigned int ElemSize;
	DType res;

	if constexpr(!std::is_floating_point<DType>::value) // case with DType = vector
	{	
		ElemSize = mean.size();
		res.resize(ElemSize);

		for(unsigned int j = 0u; j < ElemSize; ++j) // loop over each component of mean and res
		{	
			// Set lower and upper values for the uniform distribution
			typename DType::value_type a = std::max(mean[j] - sigma[j], param_genetic_algorithm.lower_bound[j]);
			typename DType::value_type b = std::min(mean[j] + sigma[j], param_genetic_algorithm.upper_bound[j]);

			// Generate the j-th random component from a normal truncated in (a,b)
			std::uniform_real_distribution<typename DType::value_type> unif_distr(normal_cdf((a - mean[j])/sigma[j]), normal_cdf((b - mean[j])/sigma[j]));

			res[j] = mean[j] + sigma[j] * probit(unif_distr(generator));
		}
	}
	else // case with DType = scalar
	{
		DType a = std::max(mean - sigma, param_genetic_algorithm.lower_bound);
		DType b = std::min(mean + sigma, param_genetic_algorithm.upper_bound);
	
		// Generate the a number from a normal truncated in (a,b)
		std::uniform_real_distribution<Real>
		unif_distr(normal_cdf(static_cast<Real>((a - mean)/sigma)), static_cast<Real>(normal_cdf((b - mean)/sigma)));
		
		res = mean + sigma * static_cast<DType>(probit(unif_distr(generator)));
		
	}

	return res;
}


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
	DType sigma;

	if constexpr(!std::is_floating_point<DType>::value) // case with DType = vector
	{
		sigma.resize(population[0].size());
		std::fill(sigma.data(), sigma.data() + sigma.size(), static_cast<typename DType::value_type>(2.0));
	}
	else // case with DType = scalar 
	{
		sigma = static_cast<DType>(2.0);
	}

	// Populate the candidate solutions in population
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i)
		population[i] = get_random_element(population[0], sigma);

	return;
}


template <class DType, class CType>
typename Genetic_Algorithm<DType, CType>::VectorXctype Genetic_Algorithm<DType, CType>::evaluation(void) const
{
	VectorXctype res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res[i] = F(population[i]);

	return res;
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::selection_and_variation(VectorXctype values)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one with a little variation of the winner 

	DType sigma1, sigma2;

	std::default_random_engine generator{std::random_device{}()};
	std::uniform_int_distribution<UInt> dice(0, 3);

	if constexpr(!std::is_floating_point<DType>::value) // case with DType = vector
	{
		sigma1.resize(population[0].size());
		std::fill(sigma1.data(), sigma1.data() + sigma1.size(), static_cast<typename DType::value_type>(0.5));

		sigma2.resize(population[0].size());
		std::fill(sigma2.data(), sigma2.data() + sigma2.size(), static_cast<typename DType::value_type>(1.5));
	}
	else // case with DType = scalar 
	{
		sigma1 = static_cast<DType>(0.5);

		sigma2 = static_cast<DType>(1.5);
	}


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


// Mutation step when DType is a vector
template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::mutation(void)
{	
	if constexpr(std::is_floating_point<DType>::value) // No mutation in scalar case
		return;
	else
	{
		std::default_random_engine generator{std::random_device{}()};

		std::uniform_real_distribution<Real> distr(0.0, 1.0);
		std::uniform_int_distribution<UInt> distr_indices(0, param_genetic_algorithm.N - 1);

		for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		{	
			if(distr(generator) < 0.25)  // mutation for 1/4 of the population on average
			{
				UInt idx1 = distr_indices(generator);
				UInt idx2 = distr_indices(generator);
				std::swap(population[idx1][0], population[idx2][0]); // swap the first component between two candidate solutions (crossover)
			}
		}

		return;
	}
}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::apply(void)
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
		++iter;

		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values);
		Rprintf("selection and variation done\n");
		mutation();
		Rprintf("mutation done\n");

		// Find the best solution of this iteration
		F_values = evaluation();
		Rprintf("evaluation done\n");

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
		
		goOn = counter < 4;



		// CHECK
		double best1 = best[0];
		double best2 = best[1];
		Rprintf("best angle and intensity %f , %f \n", best1, best2);


	}
	
	return;
}

#endif