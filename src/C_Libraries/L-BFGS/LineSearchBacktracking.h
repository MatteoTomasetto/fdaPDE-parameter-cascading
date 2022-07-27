// Copyright (C) 2016-2022 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef __LINE_SEARCH_BACKTRACKING_H__
#define __LINE_SEARCH_BACKTRACKING_H__

#include "../../FdaPDE.h"

///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
private:

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient VectorXr.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const VectorXr& xp, const VectorXr& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, VectorXr& grad, Scalar& dg, VectorXr& x)
    {
        // Decreasing and increasing factors
        const Scalar dec = 0.5;
        const Scalar inc = 2.1;

        // Check the value of step
        if (step <= Scalar(0))
        {
            Rprintf("'step' must be positive");
            step = Scalar(1.0);
        }

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > 0)
        {
            Rprintf("the moving direction increases the objective function value");
            return;
        }

        const Scalar test_decr = param.ftol * dg_init;
        Scalar width;

        int iter;
        for (iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad);

            if (fx > fx_init + step * test_decr || (fx != fx))
            {
                width = dec;
            }
            else
            {
                // Armijo condition is met
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                    break;

                const Scalar dg = grad.dot(drt);
                if (dg < param.wolfe * dg_init)
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                        break;

                    if (dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if (step < param.min_step)
            {
                Rprintf("the line search step became smaller than the minimum value allowed");
                return;
            }

            if (step > param.max_step)
            {
                Rprintf("the line search step became larger than the maximum value allowed");
                return;
            }

            step *= width;
        }

        if (iter >= param.max_linesearch)
        {
            Rprintf("the line search routine reached the maximum number of iterations");
            return;
        }
    }
};


#endif  // LBFGSPP_LINE_SEARCH_BACKTRACKING_H
