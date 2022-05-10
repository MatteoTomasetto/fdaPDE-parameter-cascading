#ifndef __OPTIMIZATION_ALGORITHM_IMP_H__
#define __OPTIMIZATION_ALGORTIHM_IMP_H__


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::initialization(void)
{

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::evaluation(void)
{

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::selection(VectorXctype values)
{

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::variation(void)
{

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::replacement(void)
{

}


template <class DType, class CType>
void Genetic_Algorithm<DType, CType>::apply(void)
{	
	initialization();

	VectorXctype F_values;

	unsigned int iter = 0u;

	while(iter < max_iter_parameter_cascading && goOn)
	{	
		++iter;

		F_values = evaluate();

		selection(F_values);

		variation();

		replacement();

	}
	
	return;
}



#endif