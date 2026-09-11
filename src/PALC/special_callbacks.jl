
"""
    FoldBifurcationTerminationCallback

A callback to terminate the continuation at a fold bifurcation. See constructor for more details.

# Fields
- `tol::Float`: Regula-falsi tolerance
- `use_det::Bool`: Use the determinant-based approach to detect folds.
"""
struct FoldBifurcationTerminationCallback <: RootSolveContinuationCallback
    tol::Float64
    use_det::Bool

    @doc"""
        FoldBifurcationTerminationCallback(; tol=1e-12, use_det=true)
    
    Constructor for `FoldBifurcationTerminationCallback`. See struct docs for details.

    Detects a fold bifurcation by monitoring the sign of the determinant of the $ n×n $ system Jacobian $∂F/∂u$,
    or by monitoring the sign of the prediction of the continuation parameter λ, i.e., determinant and bordered detection methods, respectively.
    Whenever a sign-change occurs, a regula-falsi solver attempts to precisely locate the root to tolerance `tol`.
    Upon success, the continuation is terminated.

    # Kwargs
    - `tol::Float`: Regula-falsi tolerance. Default: `1e-12`
    - `use_det::Bool`: Use the determinant-based approach to detect folds. Default `true`

    # Example Construction
    ```julia
    callback = FoldBifurcationTerminationCallback()
    ```
    """
    function FoldBifurcationTerminationCallback(; tol=1e-12, use_det=true)
        return new(tol, use_det)
    end
end

"""
    FoldBifurcationDetectionCallback

A callback to detect and save fold bifurcation points. See constructor for more info.

# Fields
- `tol::Float`: Regula-falsi tolerance
- `use_det::Bool`: Use the determinant-based approach to detect folds.
"""
struct FoldBifurcationDetectionCallback <: RootSolveContinuationCallback
    tol::Float64
    use_det::Bool
    
    @doc"""
        FoldBifurcationDetectionCallback(; tol=1e-12, use_det=true)
    
    Constructor for `FoldBifurcationTerminationCallback`. 

    Detects a fold bifurcation by monitoring the sign of the determinant of the $ n×n $ system Jacobian $∂F/∂u$,
    or by monitoring the sign of the prediction of the continuation parameter λ, i.e., determinant and bordered detection methods, respectively.
    Whenever a sign-change occurs, a regula-falsi solver attempts to precisely locate the root to tolerance `tol`.
    Upon success, the found point is saved to `PALCCache.detected_points`, if it is within the bounds `[λmin,λmax]`.

    # Kwargs
    - `tol::Float`: Regula-falsi tolerance. Defualts to `1e-12`.
    - `use_det::Bool`: Use the determinant-based approach to detect folds. Defaults to `true`

    # Examples
    ```julia
    cb1 = FoldBifurcationDetectionCallback() # callback using determinant-based fold detection
    cb2 = FoldBifurcationDetectionCallback(; use_det=false) # callback using bordered fold detection
    ```
    """
    function FoldBifurcationDetectionCallback(; tol=1e-12, use_det=true)
        return new(tol, use_det)
    end
end

# Utility function for handling the detection of fold bifurcations
function bordered_fold_detection_callback_function(u, λ, cache, alg, prob)
    # Fold detection test is based on the bordered prediction strategy. We want to
    # terminate at a point where δλ = 0.0.

    # === Fill the boardered matrix ===
    # This does the same thing as set_bordered_matrix!(...) but instead uses u and λ
    # rather than the cached uλ0
    n = length(u)
    eval_J!(cache.Jfun, cache.Ffun, u, λ, prob.f)
    cache.bordered_mat[1:n, 1:(n + 1)] .= cache.Jfun
    ddotdu1!(view(cache.bordered_mat, n + 1, 1:n), cache.δu0, alg.inner_prod)
    cache.bordered_mat[n + 1, n + 1] = ddotdλ1(cache.δλ0, alg.inner_prod)

    # Solve the bordered linear system
    # Note that ideally we'd also pass the solver cache into this function, but I don't feel
    # like further updating the code so for now we'll allocate a new solver internal to
    # this function...

    # TODO: Update code to use solver cache here
    x = begin
        ls_sol = solve(LinearProblem(cache.bordered_mat, cache.bordered_b), alg.linsolve)
        ls_sol.u
    end

    # Scale the predicted tangent
    #scale_predicted_tangent!(x, cache, alg)
    xn = sqrt(alg.inner_prod(view(x, 1:n), x[n + 1]))
    α = sign(alg.inner_prod(view(x, 1:n), cache.δu0, x[n + 1], cache.δλ0)) / xn
    x .*= α
    return x[end]
end

function determinant_fold_detection_callback_function(u, λ, cache, alg, prob)
    n = length(u)
    eval_J!(cache.Jfun, cache.Ffun, u, λ, prob.f)

    # we want to only look at the determinant of this portion, the full jacobian is always rank deficient (non-square)
    Ju = view(cache.Jfun, 1:n, 1:n) 
    return det(Ju)
end

# Handle fold bifurcation termination callback
function handle_termination_callback(cb::FoldBifurcationTerminationCallback, cache, alg, p)
    if cb.use_det
        return InternalTerminateContinuationCallback(
            determinant_fold_detection_callback_function, cache, alg, p; tol=cb.tol
        )
    else
        return InternalTerminateContinuationCallback(
            bordered_fold_detection_callback_function, cache, alg, p; tol=cb.tol
        )
    end
end

# Handle fold detection callback
function handle_detection_callback(cb::FoldBifurcationDetectionCallback, cache, alg, p)
    if cb.use_det
        return InternalDetectionCallback(
            determinant_fold_detection_callback_function, cache, alg, p, :Fold; tol=cb.tol
        )
    else
        return InternalDetectionCallback(
            bordered_fold_detection_callback_function, cache, alg, p, :Fold; tol=cb.tol
        )
    end
end