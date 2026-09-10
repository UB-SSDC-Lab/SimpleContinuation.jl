
function continuation(
    p::ContinuationProblem,
    alg::PALC;
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-6,
    dsmax=1.0,
    initial_tangent=SecantInitialTangent(),
    max_cont_steps=1000,
    newton_iter=10,
    newton_tol=1e-10,
    newton_max_resid=1.0,
    term_callback=nothing,
    analysis_callback=nothing,
    detection_callback=nothing,
    trace=Silent(),
    save_verbose=false,
    step_limiter=nothing
)
    # Construct PALC Cache
    cache = PALCCache(p, alg, ds0)

    # Wrap user provided callback functions
    handled_term_callback = handle_termination_callback(term_callback, cache, alg, p)
    handled_detection_callback = handle_detection_callback(detection_callback, cache, alg, p)

    # Construct numerical method cache
    solvers = PALCSolverCache(p, alg, cache, newton_iter, newton_tol, newton_max_resid)

    # Initialize continuation
    initialize_palc!(initial_tangent, cache, alg, p, solvers, trace, dsmin, dsmax)

    # Continuation loop
    continuation!(
        cache,
        alg,
        p,
        solvers,
        dsmin,
        dsmax,
        max_cont_steps,
        handled_term_callback,
        analysis_callback,
        handled_detection_callback,
        trace,
        save_verbose,
        step_limiter
    )
    if both_sides
        prepare_continuation_in_reverse_direction!(cache, ds0)
        continuation!(
            cache,
            alg,
            p,
            solvers,
            dsmin,
            dsmax,
            max_cont_steps,
            handled_term_callback,
            analysis_callback,
            handled_detection_callback,
            trace,
            save_verbose,
            step_limiter
        )
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
    term_callback,
    analysis_callback,
    detection_callback,
    trace,
    save_verbose,
    step_limiter
)
    # Initialize the callback(s)
    initialize!(term_callback, cache, alg, p)
    initialize!(detection_callback, cache, alg, p)

    # Continuation loop
    iter = 0
    done = false
    success = false
    while !done
        iter += 1

        # Perform prediction step
        palc_prediction!(cache, alg, p, solvers, trace)

        # Perform correction step
        success, terminate_continuation = palc_correction!(
            cache, alg, p, solvers, dsmin, dsmax, term_callback, analysis_callback, detection_callback, step_limiter, trace
        )

        # Save addtl info
        success && save_verbose_step_info!(cache, save_verbose)

        if iter >= max_cont_steps
            done = true
            cache.ret = :Maxiters
        elseif !success # If correction step failed, end the process
            done = true
            # cache ret should be set within correction to determine failure condition (currently this is limited to min stepsize)
        elseif success && terminate_continuation # termination is due to callback or hitting bound, consider successful
            done = true
            # cache ret should be set within correction to determine termination condition
        end
    end
    return nothing
end

function save_verbose_step_info!(cache, save_verbose)
    
    if save_verbose
        push!(cache.tangents, copy(cache.δuλ0))
        push!(cache.predictions, copy(cache.uλpred))
        push!(cache.dss, cache.ds)
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
