#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_K()
{
	// Vectors to store the optimal values for each lambda in lambdas_
	VectorXr angles(lambdas_.size() + 1);
	VectorXr intensities(lambdas_.size() + 1);

	// Initialization
	angles(0) = angle_;
	intensities(0) = intensity_;

	// Vectors to store the GCV values for each lambda in lambdas_
	VectorXr GCVs(lambdas_.size());

	for (Uint iter = 0; iter < lambdas_.size(); ++iter)
	{
		// Optimization step
		// [angles(iter + 1), intensities(iter + 1)] = OPT_ALGORITHM_K(angles(iter), intensities(iter), lambdas_(iter)); //TODO

		// Compute GCV with the new parameters
		H_.set_K(angles(iter + 1), intensities(iter + 1));
		//GCVs(i) = H_.get_carrier() -> compute_GCV(); //TODO
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCVs.maxCoeff(&min_GCV_pos);

	angle_ = angles(min_GCV_pos + 1); // GCVs is shorter than angles due to initialization => index shifted
	intensity_ = intensities(min_GCV_pos + 1); // GCVs is shorter than intensities due to initialization => index shifted

	H_.set_K(angle_, intensity_);
	
	// Compute increment
	increment += std::sqrt((angle_ - anlges(0))*(angle_ - anlges(0)));
	increment += std::sqrt((intensity_ - intensities(0))*(intensity_ - intensities(0)));
	
	return;
};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_b()
{
	// Vectors to store the optimal values for each lambda in lambdas_
	VectorXr b1_values(lambdas_.size() + 1);
	VectorXr b2_values(lambdas_.size() + 1);

	// Initialization
	b1_values(0) = b1_;
	b2_values(0) = b2_;

	// Vectors to store the GCV values for each lambda in lambdas_
	VectorXr GCVs(lambdas_.size());

	for (Uint iter = 0; iter < lambdas_.size(); ++iter)
	{
		// Optimization step
		// [b1_values(iter + 1), b2_values(iter + 1)] = OPT_ALGORITHM_b(b1_values(iter), b2_values(iter), lambdas_(iter)); //TODO

		// Compute GCV with the new parameters
		H_.set_b(b1_values(iter + 1), b2_values(iter + 1));
		//GCVs(i) = H_.get_carrier() -> compute_GCV(); //TODO
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCVs.maxCoeff(&min_GCV_pos);

	b1_ = b1_values(min_GCV_pos + 1); // GCVs is shorter than b1_values due to initialization => index shifted
	b2_ = b2_values(min_GCV_pos + 1); // GCVs is shorter than b2_values due to initialization => index shifted

	H_.set_b(b1_, b2_);
	
	// Compute increment
	increment += std::sqrt((b1_ - b1_values(0))*(b1_ - b1_values(0)));
	increment += std::sqrt((b2_ - b2_values(0))*(b2_ - b2_values(0)));
	
	return;
};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_c()
{

	// Vector to store the optimal values for each lambda in lambdas_
	VectorXr c_values(lambdas_.size() + 1);

	// Initialization
	c_values(0) = c_;

	// Vectors to store the GCV values for each lambda in lambdas_
	VectorXr GCVs(lambdas_.size());

	for (Uint iter = 0; iter < lambdas_.size(); ++iter)
	{
		// Optimization step
		// c_values(iter + 1) = OPT_ALGORITHM_c(c_values(iter), lambdas_(iter)); //TODO

		// Compute GCV with the new parameters
		H_.set_c(c_values(iter + 1));
		//GCVs(i) = H_.get_carrier() -> compute_GCV(); //TODO
		
		iter += 1;
	}

	// Find the minimum GCV and save the related parameters
	Uint min_GCV_pos;
	GCVs.maxCoeff(&min_GCV_pos);

	c_ = c_values(min_GCV_pos + 1); // GCVs is shorter than c_values due to initialization => index shifted
	
	H_.set_c(c_);
	
	// Compute increment
	increment += std::sqrt((c_ - c_values(0));
	
	return;
};


template <typename ... Extensions>
bool Parameter_Cascading<... Extension>::apply()
{
	for(Uint iter = 0; iter < max_iter_parameter_cascading_ && goOn; ++iter)
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
	
	return (iter < max_iter_parameter_cascading_);

};

#endif
