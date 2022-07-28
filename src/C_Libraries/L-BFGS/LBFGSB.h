// Copyright (C) 2020-2022 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license
//
// Documentation at https://lbfgspp.statr.me/
//
//
// Simple example of usage:
//
//      Eigen::Vector2d x(1.0, 1.0); // set starting point
//      LBFGSBParam<Real> param;
//      LBFGSBSolver<Real> solver(param);
//      Real fx; // vairable to store min value of F 
//      std::function<Real (Eigen::Vector2d)> F = [](Eigen::Vector2d x){return x(0) * x(0) + x(1) * x(1);}; // function to minimize
//      std::function<Eigen::Vector2d (Eigen::Vector2d)> dF = [](Eigen::Vector2d x) // gradient 
//      {   Eigen::Vector2d res;
//          res << 2 * x(0), 2 * x(1);
//          return res;
//      }
//
//      UInt niter = solver.minimize(F, dF, x, fx, lower_bound, upper_bound); // x and fx will be directly modified inside the function
// Notice that the original implementation of "minimize" requires a single function in input where the gradient is modified by reference,
// Instead now F and dF are passed to the function 
 
#ifndef __LBFGSB_H__
#define __LBFGSB_H__

#include <vector>
#include "Param.h"
#include "BFGSMat.h"
#include "Cauchy.h"
#include "SubspaceMin.h"
#include "LineSearchMoreThuente.h"
#include "../../FdaPDE.h"

///
/// L-BFGS-B solver for box-constrained numerical optimization
///
template <typename Scalar, template <class> class LineSearch = LineSearchMoreThuente>
class LBFGSBSolver
{
private:
    typedef Eigen::Map<VectorXr> MapVec;
    typedef std::vector<int> IndexSet;

    const LBFGSBParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar, true> m_bfgs;        // Approximation to the Hessian matrix
    VectorXr m_fx;                         // History of the objective function values
    VectorXr m_xp;                         // Old x
    VectorXr m_grad;                       // New gradient
    Scalar m_projgnorm;                  // Projected gradient norm
    VectorXr m_gradp;                      // Old gradient
    VectorXr m_drt;                        // Moving direction

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

    // Project the vector x to the bound constraint set
    static void force_bounds(VectorXr& x, const VectorXr& lb, const VectorXr& ub)
    {
        x.noalias() = x.cwiseMax(lb).cwiseMin(ub);
    }

    // Norm of the projected gradient
    // ||P(x-g, l, u) - x||_inf
    static Scalar proj_grad_norm(const VectorXr& x, const VectorXr& g, const VectorXr& lb, const VectorXr& ub)
    {
        return ((x - g).cwiseMax(lb).cwiseMin(ub) - x).cwiseAbs().maxCoeff();
    }

    // The maximum step size alpha such that x0 + alpha * d stays within the bounds
    static Scalar max_step_size(const VectorXr& x0, const VectorXr& drt, const VectorXr& lb, const VectorXr& ub)
    {
        const int n = x0.size();
        Scalar step = std::numeric_limits<Scalar>::infinity();

        for (int i = 0; i < n; i++)
        {
            if (drt[i] > Scalar(0))
            {
                step = std::min(step, (ub[i] - x0[i]) / drt[i]);
            }
            else if (drt[i] < Scalar(0))
            {
                step = std::min(step, (lb[i] - x0[i]) / drt[i]);
            }
        }

        return step;
    }

public:
    ///
    /// Constructor for the L-BFGS-B solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSBSolver(const LBFGSBParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function subject to box constraints, using the L-BFGS-B algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f_  A function object such that `f(x)` returns the
    ///           objective function value at `x`
    /// \param df_ A function that compute the gradient of 'f' in 'x'
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    /// \param lb Lower bounds for `x`.
    /// \param ub Upper bounds for `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename FunType, typename dFunType>
    inline int minimize(FunType& f_, dFunType& df_, VectorXr& x, Scalar& fx, const VectorXr& lb, const VectorXr& ub)
    {
        std::function<Scalar (const VectorXr&, VectorXr&)> f = [&f_, &df_](const VectorXr& x, VectorXr& grad)
        {   
            grad = df_(x);
            return f_(x);
        };

        // Dimension of the vector
        const int n = x.size();
        if (lb.size() != n || ub.size() != n)
        {
            Rf_error("'lb' and 'ub' must have the same size as 'x'");
        }

        // Check whether the initial vector is within the bounds
        // If not, project to the feasible set
        force_bounds(x, lb, ub);

        // Initialization
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        m_projgnorm = proj_grad_norm(x, m_grad, lb, ub);
        if (fpast > 0)
            m_fx[0] = fx;

        // std::cout << "x0 = " << x.transpose() << std::endl;
        // std::cout << "f(x0) = " << fx << ", ||proj_grad|| = " << m_projgnorm << std::endl << std::endl;

        // Early exit if the initial x is already a minimizer
        if (m_projgnorm <= m_param.epsilon || m_projgnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Compute generalized Cauchy point
        VectorXr xcp(n), vecc;
        IndexSet newact_set, fv_set;
        Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);

        /* VectorXr gcp(n);
        Scalar fcp = f(xcp, gcp);
        Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
        std::cout << "xcp = " << xcp.transpose() << std::endl;
        std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl; */

        // Initial direction
        m_drt.noalias() = xcp - x;
        m_drt.normalize();
        // Tolerance for s'y >= eps * (y'y)
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        // s and y vectors
        VectorXr vecs(n), vecy(n);
        // Number of iterations used
        int k = 1;
        for (;;)
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;
            Scalar dg = m_grad.dot(m_drt);

            // Maximum step size to make x feasible
            Scalar step_max = max_step_size(x, m_drt, lb, ub);
            
            // In some cases, the direction returned by the subspace minimization procedure
            // in the previous iteration is pathological, leading to issues such as
            // step_max~=0 and dg>=0. If this happens, we use xcp-x as the search direction,
            // and reset the BFGS matrix. This is because xsm (the subspace minimizer)
            // heavily depends on the BFGS matrix. If xsm is corrupted, then we may suspect
            // there is something wrong in the BFGS matrix, and it is safer to reset the matrix.
            // In contrast, xcp is obtained from a line search, which tends to be more robust
            if (dg >= Scalar(0) || step_max <= m_param.min_step)
            {
               // Reset search direction
                m_drt.noalias() = xcp - x;
                // Reset BFGS matrix
                m_bfgs.reset(n, m_param.m);
                // Recompute dg and step_max
                dg = m_grad.dot(m_drt);
                step_max = max_step_size(x, m_drt, lb, ub);
            }

            // Line search to update x, fx and gradient
            step_max = std::min(m_param.max_step, step_max);
            Scalar step = Scalar(1);
            step = std::min(step, step_max);
            LineSearch<Scalar>::LineSearch(f, m_param, m_xp, m_drt, step_max, step, fx, m_grad, dg, x);

            // New projected gradient norm
            m_projgnorm = proj_grad_norm(x, m_grad, lb, ub);

            /* std::cout << "** Iteration " << k << std::endl;
            std::cout << "   x = " << x.transpose() << std::endl;
            std::cout << "   f(x) = " << fx << ", ||proj_grad|| = " << m_projgnorm << std::endl << std::endl; */

            // Convergence test -- gradient
            if (m_projgnorm <= m_param.epsilon || m_projgnorm <= m_param.epsilon_rel * x.norm())
            {
                return k;
            }
            // Convergence test -- objective function value
            if (fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                if (k >= fpast && std::abs(fxd - fx) <= m_param.delta * std::max(std::max(std::abs(fx), std::abs(fxd)), Scalar(1)))
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
            vecs.noalias() = x - m_xp;
            vecy.noalias() = m_grad - m_gradp;
            if (vecs.dot(vecy) > eps * vecy.squaredNorm())
                m_bfgs.add_correction(vecs, vecy);

            force_bounds(x, lb, ub);
            Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);

            /*VectorXr gcp(n);
            Scalar fcp = f(xcp, gcp);
            Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
            std::cout << "xcp = " << xcp.transpose() << std::endl;
            std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl;*/

            SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, xcp, m_grad, lb, ub,
                                                   vecc, newact_set, fv_set, m_param.max_submin, m_drt);

            /*VectorXr gsm(n);
            Scalar fsm = f(x + m_drt, gsm);
            Scalar projgsmnorm = proj_grad_norm(x + m_drt, gsm, lb, ub);
            std::cout << "xsm = " << (x + m_drt).transpose() << std::endl;
            std::cout << "f(xsm) = " << fsm << ", ||proj_grad|| = " << projgsmnorm << std::endl << std::endl;*/

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
    /// Returning the infinity norm of the final projected gradient.
    /// The projected gradient is defined as \f$P(x-g,l,u)-x\f$, where \f$P(v,l,u)\f$ stands for
    /// the projection of a vector \f$v\f$ onto the box specified by the lower bound vector \f$l\f$ and
    /// upper bound vector \f$u\f$.
    ///
    Scalar final_grad_norm() const { return m_projgnorm; }
};


#endif
