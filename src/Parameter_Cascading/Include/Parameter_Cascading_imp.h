#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_K()
{
	// Vectors to store the optimal values for each lambda in lambdas
	VectorXr angles(lambdas.size() + 1);
	VectorXr intensities(lambdas.size() + 1);

	// Initialization
	angles(0) = angle;
	intensities(0) = intensity;

	// Vectors to store the GCV values for each lambda in lambdas
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		// [angles(iter + 1), intensities(iter + 1)] = OPT_ALGORITHM_K(angles(iter), intensities(iter), lambdas(iter)); //TODO

		// Compute GCV with the new parameters
		H.set_K(angles(iter + 1), intensities(iter + 1));
		H.get_solver().update_parameters(lambdas(iter));
		GCV_values(iter) = H.get_solver().compute_f(lambdas(iter)); // TODO WE NEED OPTIMAL GCV AND NOT GCV(lambdas_(iter))!!!
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.maxCoeff(&min_GCV_pos);

	angle = angles(min_GCV_pos + 1); // GCV_values is shorter than angles due to initialization => index shifted
	intensity = intensities(min_GCV_pos + 1); // GCV_values is shorter than intensities due to initialization => index shifted

	H.set_K(angle, intensity);
	
	// Compute increment
	increment += std::sqrt((angle - anlges(0))*(angle - anlges(0)));
	increment += std::sqrt((intensity - intensities(0))*(intensity - intensities(0)));
	
	return;
}

template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_b()
{
	// Vectors to store the optimal values for each lambda in lambdas
	VectorXr b1_values(lambdas.size() + 1);
	VectorXr b2_values(lambdas.size() + 1);

	// Initialization
	b1_values(0) = b1;
	b2_values(0) = b2;

	// Vectors to store the GCV values for each lambda in lambdas_
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		// [b1_values(iter + 1), b2_values(iter + 1)] = OPT_ALGORITHM_b(b1_values(iter), b2_values(iter), lambdas(iter)); //TODO

		// Compute GCV with the new parameters
		H.set_b(b1_values(iter + 1), b2_values(iter + 1));
		H.get_solver().update_parameters(lambdas(iter));
		GCV_values(iter) = H.get_solver().compute_f(lambdas(iter)); // TODO WE NEED OPTIMAL GCV AND NOT GCV(lambdas_(iter))!!!
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.maxCoeff(&min_GCV_pos);

	b1 = b1_values(min_GCV_pos + 1); // GCV_values is shorter than b1_values due to initialization => index shifted
	b2 = b2_values(min_GCV_pos + 1); // GCV_values is shorter than b2_values due to initialization => index shifted

	H.set_b(b1, b2);
	
	// Compute increment
	increment += std::sqrt((b1 - b1_values(0))*(b1 - b1_values(0)));
	increment += std::sqrt((b2 - b2_values(0))*(b2 - b2_values(0)));
	
	return;
}

template <typename InputCarrier>
void Parameter_Cascading<InputCarrier>::step_c()
{

	// Vector to store the optimal values for each lambda in lambdas
	VectorXr c_values(lambdas.size() + 1);

	// Initialization
	c_values(0) = c;

	// Vectors to store the GCV values for each lambda in lambdas
	VectorXr GCV_values(lambdas.size());

	for (Uint iter = 0; iter < lambdas.size(); ++iter)
	{
		// Optimization step
		// c_values(iter + 1) = OPT_ALGORITHM_c(c_values(iter), lambdas(iter)); //TODO

		// Compute GCV with the new parameters
		H_.set_c(c_values(iter + 1));
		H.get_solver().update_parameters(lambdas(iter));
		GCV_values(iter) = H.get_solver().compute_f(lambdas(iter)); // TODO WE NEED OPTIMAL GCV AND NOT GCV(lambdas_(iter))!!!
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCV_values.maxCoeff(&min_GCV_pos);

	c = c_values(min_GCV_pos + 1); // GCV_values is shorter than c_values due to initialization => index shifted
	
	H.set_c(c);
	
	// Compute increment
	increment += std::sqrt((c - c_values(0));
	
	return;
}


template <typename InputCarrier>
bool Parameter_Cascading<InputCarrier>::apply()
{
	for(Uint iter = 0; iter < max_iter_parameter_cascading && goOn; ++iter)
	{	
		iter += 1;
		
		increment = 0.0;
		
		if(update_K)
			step_K();
		
		if(update_b)
			step_b();
			
		if(update_c)
			step_c();
		
		goOn = increment > tol_parameter_cascading;
		
	}
	
	// Is a final update_param needed? or it is enough to set the optimal parameter as done in step_...() ?
		
	return (iter < max_iter_parameter_cascading_);

}

#endif
