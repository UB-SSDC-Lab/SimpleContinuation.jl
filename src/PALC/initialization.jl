
function initialize_palc!(cache::PALCCache, alg::PALC, p::ContinuationProblem, solvers, trace::AbstractTraceLevel)
    # Get incormation from the problem
    u0 = p.u0
    λ0 = p.λ0

    # Make sure our current iterate is the initial guess provided by user
    set_successful_iterate!(cache, u0, λ0, false)

    # Set natural continuation parameter
    set_natural_continuation_parameter!(cache, λ0)

    # Print trace if desired
    print_initializaiton_trace(cache, trace, 1)

    # Solve initial problem with user provided guess
    usol, retcode = solve_natural_nlp!(solvers, u0, trace)

    # Update current iterate if solve successfull, otherwise error
    if SciMLBase.successful_retcode(retcode)
        set_successful_iterate!(cache, usol, λ0, true)
    else
        error("Initial solve failed with user provided guess!")
    end

    # Perturb the natural continuation parameter to construct initial tangent
    # with secant method
    λpert = alg.ϵλ * cache.ds
    perturb_natural_continuation_parameter!(cache, λpert)

    # Print trace if desired
    print_initializaiton_trace(cache, trace, 2)

    # Resolve the problem with perturbed parameter
    usol, retcode = solve_natural_nlp!(solvers, usol, trace)

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

        update_tangent!(cache, δuλ0)        # Update the tangent direction in cache
        cache.δuλ0_initial .= δuλ0          # Save initial tangent
    else
        error("Solve to compute initial tangent failed! Consider reducing perturbation size.")
    end
    return nothing
end

function print_initializaiton_trace(cache::PALCCache, trace::Silent, stage::Int)
    return nothing
end
function print_initializaiton_trace(cache::PALCCache, trace::NonSilentTraceLevel, stage::Int)
    if stage == 1
        println("Initializing PALC Algorithm:")
        println("  Solving with provided guess: λ = $(cache.λn)")
    elseif stage == 2
        println("  Computing initial tangent:   λ = $(cache.λn)")
    end
    return nothing
end