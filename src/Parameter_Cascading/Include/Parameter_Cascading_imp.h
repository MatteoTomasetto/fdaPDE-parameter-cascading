#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_K()
{

	// PSEUDOCODE:	
	
	// Kold = K;
	// vector<Kappas> vK;
	// vK[0] = K;
	// vector<Real> = vGCV;
	
	// for i=1:lambdas_.size()
	//		vK[i] = OPT_ALGORITHM(vK[i-1], lambdas_[i], ...);
	//		set vK[i] in H_ -> Carrier -> get.regressionData()
	//		vGCV[i] = H_ -> Carrier -> compute_GCV();
	
	// j = argmin(vGCV);
	// K = vK[j];
	// set K in H_ -> Carrier -> get.regressionData(). This will be used in the further steps
	
	// if(computeError(Kold, K) < tol_PDE_PARAM)
	//		update_K = false;
	
	return;

};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_b()
{

	// PSEUDOCODE:	
	
	// bold = b;
	// vector<b> vb;
	// vb[0] = b;
	// vector<Real> = vGCV;
	
	// for i=1:lambdas_.size()
	//		vb[i] = OPT_ALGORITHM(vb[i-1], lambdas_[i], ...);
	//		set vb[i] in H_ -> Carrier -> get.regressionData()
	//		vGCV[i] = H_ -> Carrier -> compute_GCV();
	
	// j = argmin(vGCV);
	// b = vb[j];
	// set b in H_ -> Carrier -> get.regressionData(). This will be used in the further steps
	
	// if(computeError(b, bold) < tol_PDE_PARAM)
	//		update_b = false;


};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_c()
{


	// PSEUDOCODE:	
	
	// cold = c;
	// vector<c> vc;
	// vc[0] = c;
	// vector<Real> = vGCV;
	
	// for i=1:lambdas_.size()
	//		vc[i] = OPT_ALGORITHM(vc[i-1], lambdas_[i], ...);
	//		set vc[i] in H_ -> Carrier -> get.regressionData()
	//		vGCV[i] = H_ -> Carrier -> compute_GCV();
	
	// j = argmin(vGCV);
	// c = vc[j];
	// set c in H_ -> Carrier -> get.regressionData(). This will be used in the further steps
		
	// if(computeError(c, cold) < tol_PDE_PARAM)
	//		update_c = false;

};


template <typename ... Extensions>
void Parameter_Cascading<... Extension>::apply()
{


	// PSEUDOCODE:	
	
	// for i=0:max_iter_parameter_cascading && (update_k = true or update_b = true or update_c = true) 
	// 		if(update_k) step_K();
	//		if(update_b) step_b();
	//		if(update_c) step_c();
	

};

#endif
