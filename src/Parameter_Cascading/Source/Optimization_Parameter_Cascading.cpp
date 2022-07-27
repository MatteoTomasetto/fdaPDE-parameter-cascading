#include "../Include/Optimization_Parameter_Cascading.h"
#include <array>
#include <algorithm>
#include <type_traits>

// Generate a random VectorXr element from a truncated normal centered in "mean" 
// with standard deviation sigma
VectorXr Genetic_Algorithm::get_random_element(const VectorXr& mean, const Real& sigma)
{
	std::default_random_engine generator(seed++);

 	unsigned int ElemSize = mean.size();
 	VectorXr res(ElemSize);
 	
 	// Loop over each component of mean and res and generate random the components of res
	for(unsigned int j = 0u; j < ElemSize; ++j)
 	{	
 		// Generate the j-th random component from a normal truncated wrt lower and upper bound in param_genetic_algorithm
 		std::uniform_real_distribution<Real>
 		unif_distr(normal_cdf((param_genetic_algorithm.lower_bound[j] - mean[j])/sigma),
 				   normal_cdf((param_genetic_algorithm.upper_bound[j] - mean[j])/sigma));

 		res[j] = mean[j] + sigma * probit(unif_distr(generator));
 	}
	
 	return res;
}
			 

// Cumulative distribution function of a standard normal (phi function)
Real Genetic_Algorithm::normal_cdf(const Real& x) const
{
	return 0.5 + 0.5 * std::erf(x * M_SQRT1_2);
}

// Inverse cumulative distribution function of a standard normal (probit function)
// approximated via Beasley-Springer-Moro algorithm.
Real Genetic_Algorithm::probit(const Real& u) const
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

void Genetic_Algorithm::initialization(void)
{	
	Real sigma = 2.5; // Not small standard deviation to explore enough a region near population[0]

	// Populate the candidate solutions in population
	for(unsigned int i = 1u; i < param_genetic_algorithm.N; ++i){
		population[i] = get_random_element(population[0], sigma);
	}

	return;
}


typename Genetic_Algorithm::evalType Genetic_Algorithm::evaluation(void) const
{
	evalType res(param_genetic_algorithm.N);

	for(unsigned int i = 0u; i < param_genetic_algorithm.N; ++i)
		res[i] = F(population[i]); // Evaluate F in each candidate solution in population

	return res;
}


void Genetic_Algorithm::selection_and_variation(evalType values, Real alpha)
{
	// Binary tournament selection: for each couple in population we keep the best one (winner) in terms of loss function
	// Then we replace the worst one in 2 different ways:
	// 		a) winner + gaussian noise
	//		b) best + gaussian noise
	// alpha is a parameter that increases iteration by iteration of the algorithm.
	// The standard deviation of the gaussian noise decreases wrt alpha to concentrate more and more the candidate solutions near optimal values
	// The probability of performing the option (b) increases as alpha increases (i.e. we focus more and more on "best")

	std::default_random_engine generator(seed++);

	std::uniform_int_distribution<UInt> dice(0,  static_cast<UInt>(alpha)); // Higher probability to consider "best" below as alpha increases 

	Real adapt_sigma = 2.0 / std::log(alpha + 2.0); // Smaller standard deviation iteration by iteration (as alpha increases)

	for (unsigned int i = 0u; i < param_genetic_algorithm.N - 1; i += 2u)
	{	
		unsigned int idx_loser = values[i] > values[i+1] ? i : i + 1;
		unsigned int idx_winner = idx_loser == i ? i + 1 : i;
		UInt choice = dice(generator);
		
		if(choice == 0 || choice == 1)
			population[idx_loser] = get_random_element(population[idx_winner], adapt_sigma);
		else
			population[idx_loser] = get_random_element(best, adapt_sigma);
	}

	return;
}

void Genetic_Algorithm::apply(void)
{	
	Rprintf("Start genetic algorithm\n");

	initialization();
	
	// Evaluate the loss function for population elements: this is needed for selection process
	evalType F_values(param_genetic_algorithm.N);
	F_values = evaluation();
	
	unsigned int iter = 0u;
	unsigned int counter = 0u; // counter how many times "best" does not change

	while(iter < max_iterations_genetic_algorithm && goOn)
	{	
		// Genetic algorithm steps to modify the population (keep most promising candidate + generate new candidate solutions)
		selection_and_variation(F_values, static_cast<Real>(iter));
		
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

		++iter;

		Rprintf("Current min value reached: %f\n", min_value);
		
}

Rprintf("Final min value reached: %f\n", min_value);

Rprintf("End genetic algorithm\n");
	
	return;
}

void Gradient_Descent_fd::upgrade_best(void)
{	
 	VectorXr new_best = best;

	// Set some useful parameters
	Real alpha = 1.0;
	Real beta = 0.5;
	Real f = F(best);
	VectorXr df = dF(best);
	Real df_norm = df.squaredNorm();

	// Update best
	new_best = best - alpha * df;

 	// Check bounds
 	for(unsigned int i = 0u; i < best.size(); ++i)
 	{
 		// Exploit periodicity if present
 		if(param_gradient_descent_fd.periods[i] != 0.0)
 		{
			while(new_best[i] < param_gradient_descent_fd.lower_bound[i])
					new_best[i] += param_gradient_descent_fd.periods[i];
			while(new_best[i] > param_gradient_descent_fd.upper_bound[i])
					new_best[i] -= param_gradient_descent_fd.periods[i];
 		}

 		else
 		{	
			// Check upper and lower bounds
 			while(new_best[i] < param_gradient_descent_fd.lower_bound[i] || new_best[i] > param_gradient_descent_fd.upper_bound[i])
 			{	
 				new_best[i] = best[i];			 				
 				alpha *= beta; // Re-upgrade best with a smaller alpha in order to remain inside the bounds
 				new_best[i] -= alpha*df[i];
 			}
 		}
	}

	// Backtracking line search to set the step size
	UInt counter = 0;
	while(F(new_best) > f - alpha * df_norm / 2.0 || counter < 40)
	{
		alpha *= beta;
		new_best = best - alpha * df;
		counter++;
	}

	best = new_best;

 	return;
}

void Gradient_Descent_fd::apply(void)
{	
	Rprintf("Start gradient descent algorithm\n");

	unsigned int iter = 0u;
	VectorXr old_sol = best;
	Real increment = 0.0;

	while(iter < max_iterations_gradient_descent_fd && goOn)
	{	
		++iter;

		upgrade_best();

		increment = (best - old_sol).norm();
		
		old_sol = best;

		goOn = (increment < tol_gradient_descent_fd) ? false : true;

	}

	Rprintf("End gradient descent algorithm\n");

	return;
}
