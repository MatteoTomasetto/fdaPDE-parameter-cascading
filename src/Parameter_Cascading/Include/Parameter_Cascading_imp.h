#ifndef __PARAMETER_CASCADING_IMP_H__
#define __PARAMETER_CASCADING_IMP_H__

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_K()
{

	// PSEUDOCODE:	
	
	// ... Kold = K;
	// vector<Kappas> vK;
	// vK[0] = K;
	// vector<Real> = vGCV;
	
	// for i=0:lambdas_.size()
	//		vK[i] = OPT_ALGORITHM(vK[i-1], lambdas_[i], ...);
	//		vGCV[i] = compute_GCV(vK[i]);
	
	// j = argmin(vGCV);
	// K = vK[j];
	
	// if(computeError(K, Kold) < tol_PDE_PARAM)
	//		update_K = false;
	
	return;

};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_b()
{

	// PSEUDOCODE:	
	
	// ... bold = b;
	// vector<b> vb;
	// vb[0] = b;
	// vector<Real> = vGCV;
	
	// for i=0:lambdas_.size()
	//		vb[i] = OPT_ALGORITHM(vb[i-1], lambdas_[i], ...);
	//		vGCV[i] = compute_GCV(vb[i]);
	
	// j = argmin(vGCV);
	// b = vb[j];
	
	// if(computeError(b, bold) < tol_PDE_PARAM)
	//		update_b = false;


};

template <typename ... Extensions>
void Parameter_Cascading<... Extension>::step_c()
{


	// PSEUDOCODE:	
	
	// ... cold = c;
	// vector<c> vc;
	// vc[0] = c;
	// vector<Real> = vGCV;
	
	// for i=0:lambdas_.size()
	//		vc[i] = OPT_ALGORITHM(vc[i-1], lambdas_[i], ...);
	//		vGCV[i] = compute_GCV(vc[i]);
	
	// j = argmin(vGCV);
	// c = vc[j];
	
	// if(computeError(c, cold) < tol_PDE_PARAM)
	//		update_c = false;

};

#endif
