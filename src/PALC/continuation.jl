
function continuation(
    p::ContinuationProblem, alg::PALC;
    both_sides      = false,
    ds0             = 1e-2,
    dsmin           = 1e-6,
    dsmax           = 1.0,
    max_cont_steps  = 1000,
    newton_iter     = 10,
    newton_tol      = 1e-10,
    verbose         = 0,
)
    # Construct PALC Cache
    cache = PALCCache(p, alg, ds0)

    # Construct numerical method cache
    solvers = PALCSolverCache(p, alg, cache, newton_iter, newton_tol)

    # Initialize continuation
    initialize_palc!(cache, alg, p, solvers)

    # Continuation loop
    continuation!(cache, alg, p, solvers, dsmin, dsmax, max_cont_steps, verbose)
    if both_sides
        prepare_continuation_in_reverse_direction!(cache, ds0)
        continuation!(cache, alg, p, solvers, dsmin, dsmax, max_cont_steps, verbose)
    end

    return cache
end

function continuation!(cache::PALCCache, alg::PALC, p::ContinuationProblem, solvers::PALCSolverCache, dsmin, dsmax, max_cont_steps, verbose)
    # Continuation loop
    iter    = 0
    done    = false
    success = false
    while !done
        iter += 1

        # Perform prediction step
        palc_prediction!(cache, alg, p, solvers)

        # Perform correction step
        success, hit_bnd = palc_correction!(cache, alg, p, solvers, dsmin, dsmax)

        if iter >= max_cont_steps
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

function initialize_palc!(cache::PALCCache, alg::PALC, p::ContinuationProblem, solvers::PALCSolverCache, verbose = 0)
    # Get incormation from the problem
    u0 = p.u0
    λ0 = p.λ0

    # Make sure our current iterate is the initial guess provided by user
    set_successful_iterate!(cache, u0, λ0, false)

    # Set natural continuation parameter
    set_natural_continuation_parameter!(cache, λ0)

    # Solve initial problem with user provided guess
    usol, retcode = solve_natural_nlp!(solvers, u0, verbose)

    # Update current iterate if solve successfull, otherwise error
    if SciMLBase.successful_retcode(retcode)
        set_successful_iterate!(cache, usol, λ0, true)
    else
        error("Initial solve failed with user provided guess!")
    end

    # Perturb the natural continuation parameter to construct initial tangent
    # with secant method
    λpert = 1e-6*cache.ds
    perturb_natural_continuation_parameter!(cache, λpert)

    # Resolve the problem with perturbed parameter
    usol, retcode = solve_natural_nlp!(solvers, usol, verbose)

    # Get the solution
    if SciMLBase.successful_retcode(retcode)
        # Compute and set the initial tangent
        n           = length(u0)
        u0          = cache.u0              # The current solution (from first solve)
        δuλ0        = cache.δuλ0            # The predicted tangent direction

        δuλ0[1:n]  .= usol .- cache.u0      # Setting secant direction
        δuλ0[end]   = λpert

        ninv        = 1.0 / norm(δuλ0)      # Scale secant direciton to unit vector
        δuλ0      .*= ninv

        update_tangent!(cache, δuλ0)  # Update the tangent direction in cache
        cache.δuλ0_initial .= δuλ0          # Save initial tangent
    else
        error("Solve to compute initial tangent failed! Consider reducing perturbation size.")
    end
    return nothing
end