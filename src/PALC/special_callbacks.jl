
# This callback terminates the continuation process when a fold bifurcation is detected.
struct FoldBifurcationTerminationCallback <: RootSolveContinuationCallback
    tol::Float64
    function FoldBifurcationTerminationCallback(; tol=1e-12)
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

# Incomplete - just testing. Remove later if not used
# This callback terminates the continuation process when the slope of the predicted tangent in a prescried dimension rises above a certain threshold.
# For example, I'm using this to detect rapid increases in the time of flight. This may prematurely trigger as we approach
# fold bifurcations (though in an initial test this did not happen, I think because the folds in κ-tf space are generally cusps rather than smooth)
# so a better implementation is needed. But it works for the problem I'm using it on right now, where I'm avoiding degeneracy 
# due to a rapid increase in tf rather than a fold
struct MaxSlopeTerminationCallback <: RootSolveContinuationCallback
    indexes::Vector{Int}
    threshold::Float64
    tol::Float64
    function MaxSlopeTerminationCallback(indexes,threshold; tol=1e-12)
        return new(indexes, threshold, tol)
    end
end


function max_slope_callback_function(u, λ, cache, alg, prob, threshold, indexes)
    
    # Calculate the bordered prediction in the same manner as the FoldBifurcationTerminationCallback
    n = length(u)
    eval_J!(cache.Jfun, cache.Ffun, u, λ, prob.f)
    cache.bordered_mat[1:n, 1:(n + 1)] .= cache.Jfun
    ddotdu1!(view(cache.bordered_mat, n + 1, 1:n), cache.δu0, alg.inner_prod)
    cache.bordered_mat[n + 1, n + 1] = ddotdλ1(cache.δλ0, alg.inner_prod)

    x = begin
        ls_sol = solve(LinearProblem(cache.bordered_mat, cache.bordered_b), alg.linsolve)
        ls_sol.u
    end

    # Scale the predicted tangent
    #scale_predicted_tangent!(x, cache, alg)
    xn = sqrt(alg.inner_prod(view(x, 1:n), x[n + 1]))
    α = sign(alg.inner_prod(view(x, 1:n), cache.δu0, x[n + 1], cache.δλ0)) / xn
    x .*= α

    un = view(x, 1:n)
    λn = view(x, n+1)

    # Now check if the slope (norm) for the desired indexes is greater than some threshold
    ret = norm(abs.(un[indexes])./λn)-threshold
    return ret

end

# Handle slope termination callback
function handle_termination_callback(cb::MaxSlopeTerminationCallback, cache, alg, p)
    fn(u, λ, cache, alg, prob) = max_slope_callback_function(u, λ, cache, alg, prob, cb.threshold, cb.indexes)
    return InternalTerminateContinuationCallback(
        fn, cache, alg, p; tol=cb.tol
    )
end