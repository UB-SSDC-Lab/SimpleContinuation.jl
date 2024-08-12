
function palc_correction!(cache, alg, p::ContinuationProblem, solvers, dsmin, dsmax, trace)
    # Get cache variables 
    θ       = cache.θ
    u0      = cache.u0
    λ0      = cache.λ0
    δu0     = cache.δu0
    δλ0     = cache.δλ0
    uλpred  = cache.uλpred
    n       = length(δu0)

    # Get problem variables
    λmin    = p.λ_bounds[1]
    λmax    = p.λ_bounds[2]

    # Solve nonlinear problem (reducing step-size if necessary)
    attempts = 0
    success  = true
    done     = false
    hit_bnd  = NaN
    while !done
        # Update attempts
        attempts += 1

        # Update uλpred (clamping ds to try and stay in λ bounds)
        # If clamped, set hit_bnd to the bound hit and we'll resolve
        # if successful with constant λ
        uλpred[end]  = λ0  + cache.ds*δλ0
        if uλpred[end] < λmin
            cache.ds    = (λmin - λ0) / δλ0
            uλpred[end] = λmin
            hit_bnd     = λmin
            print_correction_trace(cache, trace, 2)
        elseif uλpred[end] > λmax
            cache.ds    = (λmax - λ0) / δλ0
            uλpred[end] = λmax
            hit_bnd     = λmax
            print_correction_trace(cache, trace, 2)
        else
            print_correction_trace(cache, trace, 1)
        end
        uλpred[1:n] .= u0 .+ cache.ds.*δu0

        # Solve the palc nonlinear problem
        uλ, retcode = solve_palc_nlp!(solvers, uλpred, trace)

        # Check if successful
        if SciMLBase.successful_retcode(retcode)
            # Check if we crossed the boundary
            if uλ[end] < λmin 
                hit_bnd = λmin
            elseif uλ[end] > λmax
                hit_bnd = λmax
            end

            if isnan(hit_bnd) 
                # Push solution and set done
                set_successful_iterate!(cache, uλ) 
                done = true

                # Print trace if desired
                print_correction_trace(cache, trace, 3)
            else
                # Update cache without pushing solution to curve
                set_successful_iterate!(cache, uλ, false)

                # Print trace if desired
                print_correction_trace(cache, trace, 3)

                # Target solution on boundary
                flag = palc_target_solution_on_boundary!(cache, hit_bnd, solvers, trace)

                # If targeting solution on boundary was successful, we're done. Otherwise, reduce ds
                if flag
                    done = true
                else
                    hit_bnd  = NaN
                    scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
                end
            end
        else
            if isapprox(abs(cache.ds),dsmin)
                done    = true
                success = false
            else
                # Reduce step-size and reattempt
                scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
            end
        end
    end

    # Update ds is we were successful
    # Consider only updating is successful in < n number of attempts
    if success
        scale_and_clamp_ds!(cache, 1.2, dsmin, dsmax)
    end

    return success, !isnan(hit_bnd)
end

function print_correction_trace(cache::PALCCache, trace::Silent, stage)
    return nothing
end
function print_correction_trace(cache::PALCCache, trace::NonSilentTraceLevel, stage)
    if stage == 1
        # Compute predicted change
        δλ0 = cache.uλpred[end] - cache.λ0
        println("Beginning PALC correction: λ = $(cache.uλpred[end]) [$δλ0] (predict)")
    elseif stage == 2
        δλ0 = cache.uλpred[end] - cache.λ0
        println("Beginning PALC correction: λ = $(cache.uλpred[end]) [$δλ0] (predict - clamped to boundary)")
    elseif stage == 3
        println("Correction successful: λ = $(cache.uλ0[end])")
    elseif stage == 4
        println("Beginning natural correction: λ = $(cache.λn)")
    elseif stage == 5
        println("Natural continuation successful")
    elseif stage == 6
        println("Natural continuation failed")
    end
    return nothing
end

function scale_and_clamp_ds!(cache, scale, dsmin, dsmax)
    sign_ds = sign(cache.ds)
    abs_ds  = abs(cache.ds)
    new_ds  = clamp(scale*abs_ds, dsmin, dsmax)
    cache.ds = sign_ds*new_ds
    return nothing
end

function palc_target_solution_on_boundary!(cache, λ0, solvers, trace)
    # Set natural continuation parameter
    set_natural_continuation_parameter!(cache, λ0)

    # Print trace if desired
    print_correction_trace(cache, trace, 4)

    # Solve the natural continuation problem
    usol, retcode = solve_natural_nlp!(solvers, cache.u0, trace)

    success_flag = SciMLBase.successful_retcode(retcode)
    if success_flag
        set_successful_iterate!(cache, usol, λ0)

        # Print trace if desired
        print_correction_trace(cache, trace, 5)
    else
        # Print trace if desired
        print_correction_trace(cache, trace, 6)
    end
    return success_flag
end

# Hyperplane constraint and jacobian
function palc_hyperplane_constraint(δu, δu0, δλ, δλ0, θ, ds)
    n   = length(δu0)
    sf1 = 1.0 #θ/n
    sf2 = 1.0 #1-θ
    N = sf1*dot(δu, δu0) + sf2*δλ*δλ0 - ds
    return N
end
function palc_hyperplane_constraint_jacobian!(J, δu0, δλ0, θ)
    n = length(δu0)
    sf1 = 1.0 #θ/n
    sf2 = 1.0 #1-θ
    J[1:n] .= sf1.*δu0
    J[n+1] = sf2*δλ0
    return nothing
end

# ===== Nonlinear solve functions

# Here p is a tuple of parameters, where
#   p[1] = ContinuationFunction
#   p[2] = PALCCache
function palc_correction_function!(F, uλ, p)
    # Get parameters
    fun     = p[1]
    cache   = p[2]
    δu      = cache.δu
    δu0     = cache.δu0
    δλ0     = cache.δλ0
    θ       = cache.θ
    ds      = cache.ds

    # Get u and λ
    n = length(uλ) - 1
    u = view(uλ, 1:n)
    λ = uλ[end]

    # Evaluate the function
    eval_f!(cache.Ffun, uλ, fun)

    # Evaluate the hyperplane constraint
    δu .= u .- cache.u0
    δλ  = λ  - cache.λ0
    N   = palc_hyperplane_constraint(δu, δu0, δλ, δλ0, θ, ds)

    # Set function and return
    F[1:n] .= cache.Ffun
    F[end]  = N
    return nothing
end

function palc_correction_jacobian!(J, uλ, p)
    # Get parameters
    fun     = p[1]
    cache   = p[2]
    δu0     = cache.δu0
    δλ0     = cache.δλ0
    θ       = cache.θ
    n       = length(uλ) - 1

    # Evaluate the jacobian and set
    eval_J!(cache.Jfun, cache.Ffun, uλ, fun)
    J[1:n, :] .= cache.Jfun

    # Evaluate the hyperplane constraint jacobian
    palc_hyperplane_constraint_jacobian!(view(J, n+1, :), δu0, δλ0, θ)

    return nothing
end