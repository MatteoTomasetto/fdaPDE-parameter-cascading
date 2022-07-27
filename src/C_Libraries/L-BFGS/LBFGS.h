// Copyright (C) 2016-2022 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license
//
// Documentation at https://lbfgspp.statr.me/
//
//
// Simple example of usage:
//
//      Eigen::Vector2d x(1.0, 1.0); // set starting point
//      LBFGSParam<Real> param;
//      LBFGSSolver<Real> solver(param);
//      Real fx; // vairable to store min value of F 
//      std::function<Real (Eigen::Vector2d)> F = [](Eigen::Vector2d x){return x(0) * x(0) + x(1) * x(1);}; // function to minimize
//      std::function<Eigen::Vector2d (Eigen::Vector2d)> dF = [](Eigen::Vector2d x) // gradient 
//      {   Eigen::Vector2d res;
//          res << 2 * x(0), 2 * x(1);
//          return res;
//      }
//
//      UInt niter = solver.minimize(F, dF, x, fx); // x and fx will be directly modified inside the function
// Notice that the original implementation of "minimize" requires a single function in input where the gradient is modified by reference,
// Instead now F and dF are passed to the function 

#ifndef __LBFGS_H__
#define __LBFGS_H__

#include <Eigen/Core>
#include "Param.h"
#include "BFGSMat.h"
#include "LineSearchBacktracking.h"
#include "LineSearchBracketing.h"
#include "LineSearchNocedalWright.h"
#include "LineSearchMoreThuente.h"
#include "../../FdaPDE.h"

///
/// L-BFGS solver for unconstrained numerical optimization
///
template <typename Scalar,
          template <class> class LineSearch = LineSearchNocedalWright>
class LBFGSSolver
{
private:
    typedef Eigen::Map<VectorXr> MapVec;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar> m_bfgs;             // Approximation to the Hessian matrix
    VectorXr m_fx;                      // History of the objective function values
    VectorXr m_xp;                      // Old x
    VectorXr m_grad;                    // New gradient
    Scalar m_gnorm;                     // Norm of the gradient
    VectorXr m_gradp;                   // Old gradient
    VectorXr m_drt;                     // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if (m_param.past > 0)
            m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f_  A function object such that `f(x)` returns the
    ///           objective function value at `x`
    /// \param df_ A function that compute the gradient of 'f' in 'x'
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///

    template <typename FunType, typename dFunType>
    inline int minimize(FunType& f_, dFunType& df_, VectorXr& x, Scalar& fx)
    {
        std::function<Scalar (const VectorXr&, VectorXr&)> f = [&f_, &df_](const VectorXr& x, VectorXr& grad)
        {   
            grad = df_(x);
            return f_(x);
        };

        using std::abs;

        // Dimension of the vector
        const int n = x.size();
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        m_gnorm = m_grad.norm();
        if (fpast > 0)
            m_fx[0] = fx;

        // std::cout << "x0 = " << x.transpose() << std::endl;
        // std::cout << "f(x0) = " << fx << ", ||grad|| = " << m_gnorm << std::endl << std::endl;

        // Early exit if the initial x is already a minimizer
        if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step size
        Scalar step = Scalar(1) / m_drt.norm();

        // Number of iterations used
        int k = 1;
        for (;;)
        {
            // std::cout << "Iter " << k << " begins" << std::endl << std::endl;

            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;
            Scalar dg = m_grad.dot(m_drt);
            const Scalar step_max = m_param.max_step;

            // Line search to update x, fx and gradient
            LineSearch<Scalar>::LineSearch(f, m_param, m_xp, m_drt, step_max, step, fx, m_grad, dg, x);

            // New gradient norm
            m_gnorm = m_grad.norm();

            // std::cout << "Iter " << k << " finished line search" << std::endl;
            // std::cout << "   x = " << x.transpose() << std::endl;
            // std::cout << "   f(x) = " << fx << ", ||grad|| = " << m_gnorm << std::endl << std::endl;

            // Convergence test -- gradient
            if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.epsilon_rel * x.norm())
            {
                return k;
            }
            // Convergence test -- objective function value
            if (fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                if (k >= fpast && abs(fxd - fx) <= m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1)))
                    return k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            m_bfgs.add_correction(x - m_xp, m_grad - m_gradp);

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);
            k++;
        }

        return k;
    }

    ///
    /// Returning the gradient vector on the last iterate.
    /// Typically used to debug and test convergence.
    /// Should only be called after the `minimize()` function.
    ///
    /// \return A const reference to the gradient vector.
    ///
    const VectorXr& final_grad() const { return m_grad; }

    ///
    /// Returning the Euclidean norm of the final gradient.
    ///
    Scalar final_grad_norm() const { return m_gnorm; }
};

#endif
