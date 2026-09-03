
# This callback terminates the continuation process when a fold bifurcation is detected.
struct FoldBifurcationTerminationCallback <: RootSolveContinuationCallback
    tol::Float64
    function FoldBifurcationTerminationCallback(; tol=1e-12)
        return new(tol)
    end
end

# This callback detects folds, but does not terminate the process
# It will instead save the detected point to the cache in the detected_points field of the PALCCache
struct FoldBifurcationDetectionCallback <: RootSolveContinuationCallback
    tol::Float64
    function FoldBifurcationDetectionCallback(; tol=1e-12)
        return new(tol)
    end
end

# Utility function for handling the detection of fold bifurcations
function fold_detection_callback_function(u, λ, cache, alg, prob)
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

# Handle fold bifurcation termination callback
function handle_termination_callback(cb::FoldBifurcationTerminationCallback, cache, alg, p)
    return InternalTerminateContinuationCallback(
        fold_detection_callback_function, cache, alg, p; tol=cb.tol
    )
end

# Handle fold detection callback
function handle_detection_callback(cb::FoldBifurcationDetectionCallback, cache, alg, p)
    return InternalDetectionCallback(
        fold_detection_callback_function, cache, alg, p, :Fold; tol=cb.tol
    )
end