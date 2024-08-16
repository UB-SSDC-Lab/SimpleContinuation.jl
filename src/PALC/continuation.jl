
function continuation(
    p::ContinuationProblem, alg::PALC;
    both_sides          = false,
    ds0                 = 1e-2,
    dsmin               = 1e-6,
    dsmax               = 1.0,
    max_cont_steps      = 1000,
    newton_iter         = 10,
    newton_tol          = 1e-10,
    newton_max_resid    = 1.0,
    trace               = Silent(),
)
    # Construct PALC Cache
    cache = PALCCache(p, alg, ds0)

    # Construct numerical method cache
    solvers = PALCSolverCache(
        p, alg, cache, 
        newton_iter, 
        newton_tol, 
        newton_max_resid,
    )

    # Initialize continuation
    initialize_palc!(cache, alg, p, solvers, trace)

    # Continuation loop
    continuation!(cache, alg, p, solvers, dsmin, dsmax, max_cont_steps, trace)
    if both_sides
        prepare_continuation_in_reverse_direction!(cache, ds0)
        continuation!(cache, alg, p, solvers, dsmin, dsmax, max_cont_steps, trace)
    end

    return cache
end

function continuation!(
    cache::PALCCache, 
    alg::PALC, 
    p::ContinuationProblem, 
    solvers::PALCSolverCache, 
    dsmin, 
    dsmax, 
    max_cont_steps, 
    trace,
)
    # Continuation loop
    iter    = 0
    done    = false
    success = false
    while !done
        iter += 1

        # Perform prediction step
        palc_prediction!(cache, alg, p, solvers, trace)

        # Perform correction step
        success, hit_bnd = palc_correction!(cache, alg, p, solvers, dsmin, dsmax, trace)

        if iter >= max_cont_steps
            done = true
        elseif !success
            done = true
        elseif success && hit_bnd
            done = true
        end
    end
    return nothing
end

function prepare_continuation_in_reverse_direction!(cache::PALCCache, ds0)
    # Set arc length
    cache.ds = -ds0

    # Set current iterate
    u0 = cache.br[1][1]
    λ0 = cache.br[1][2]
    set_successful_iterate!(cache, u0, λ0, false)

    # Reset tangent (don't need to change direction because
    # the sign of ds has been changed)
    update_tangent!(cache, cache.δuλ0_initial)

    # Flip elements in br
    reverse!(cache.br)

    return nothing
end