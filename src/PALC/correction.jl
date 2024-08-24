
function palc_correction!(
    cache, alg, p::ContinuationProblem, solvers, 
    dsmin, dsmax, term_callback, analysis_callback, trace,
)
    # Get cache variables 
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
    cb_trig  = false
    rf_succ  = false
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
            # Check if callback triggered
            cb_trig = check(term_callback, uλ, cache)

            # If callback triggered, perform regula falsi root finding method and update uλ
            if cb_trig
                rf_succ = palc_target_callback_event!(uλ, cache, solvers, term_callback, trace)
                hit_bnd = NaN # Reset since we're likely not stepping as far and will recheck 
            end

            # Check if we crossed the boundary
            if uλ[end] < λmin 
                hit_bnd = λmin
            elseif uλ[end] > λmax
                hit_bnd = λmax
            end

            if cb_trig && !rf_succ # Triggered callback but rootfind was unsuccessful
                cb_trig = false
                hit_bnd = NaN
                scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
            elseif isnan(hit_bnd)
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
            if abs(cache.ds) == dsmin
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
    success && scale_and_clamp_ds!(cache, 1.2, dsmin, dsmax)

    # Update the callback if we were successfull
    success && update!(term_callback, cache)

    # Call the analysis callback if we were successful
    success && call!(analysis_callback, cache)

    # Handle termination flag
    terminate_continuation = !isnan(hit_bnd) || cb_trig

    return success, terminate_continuation
end

function print_correction_trace(cache::PALCCache, trace::Silent, stage)
    return nothing
end
function print_correction_trace(cache::PALCCache, trace::NonSilentTraceLevel, stage)
    if stage == 1
        δλ0 = cache.uλpred[end] - cache.λ0
        @printf "Beginning PALC correction: λ = %.5e [%.1e] (predict)\n" cache.uλpred[end] δλ0
    elseif stage == 2
        δλ0 = cache.uλpred[end] - cache.λ0
        @printf "Beginning PALC correction: λ = %.5e [%.1e] (predict - clamped to boundary)\n" cache.uλpred[end] δλ0
    elseif stage == 3
        # Compute angle between prediction and actual change
        θ = if length(cache.br) > 1
            δu  = cache.u_0
            δu .= cache.br[end][1] .- cache.br[end-1][1]
            δλ  = cache.br[end][2]  - cache.br[end-1][2]
            if cache.ds < 0.0
                δu .*= -1.0
                δλ   = -δλ
            end
            dp  = dot(δu, cache.δu0) + δλ*cache.δλ0
            r   = dp / (sqrt(dot(δu,δu) + δλ^2)*norm(cache.δuλ0))
            θ   = acosd(clamp(r, -1.0, 1.0))
        else
            NaN
        end

        @printf "Correction successful: λ = %.5e (θ = %3.2e°)\n" cache.uλ0[end] θ
    elseif stage == 4
        #println("Beginning natural correction: λ = $(cache.λn)")
        @printf "Beginning natural correction: λ = %.5e\n" cache.λn
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

# Function to target boundary with natural continuation
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

# Function to find when callback = 0 with regula falsi method
function palc_target_callback_event!(uλ, cache, solvers, callback, trace)
    # Get callback value at boundaries
    f_0 = callback.val_0
    f_1 = call!(callback, uλ, cache)

    # Get parameter values at boundaries
    λ_0 = cache.λ0
    λ_1 = uλ[end]

    # Get inputs at boundaries
    u_0  = cache.u_0; u_1 = cache.u_1; u_t = cache.u_t
    n    = length(uλ) - 1
    u_0 .= cache.u0
    u_1 .= view(uλ, 1:n)

    # Begin loop
    done    = false
    success = false
    while !done
        λ_2  = λ_0 - f_0*(λ_1 - λ_0) / (f_1 - f_0)
        u_t .= u_0 .+ ((λ_2 - λ_0) / (λ_1 - λ_0)).*(u_1 .- u_0)

        set_natural_continuation_parameter!(cache, λ_2)
        usol, retcode = solve_natural_nlp!(solvers, u_t, trace)

        success_flag = SciMLBase.successful_retcode(retcode)
        if success_flag
            # Call the callback function
            f_2 = call!(callback, usol, λ_2, cache)

            if abs(f_2) <= callback.tol
                # Set flags
                done    = true
                success = true

                # Update iterate
                uλ[1:n] .= usol
                uλ[end]  = λ_2
            elseif f_0*f_2 < 0
                u_1 .= usol
                λ_1  = λ_2
                f_1  = f_2
            else
                u_0 .= usol
                λ_0  = λ_2
                f_0  = f_2
            end
        else
            done = true
        end
    end

    return success
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
    alg     = p[3]
    δu      = cache.δu
    δu0     = cache.δu0
    δλ0     = cache.δλ0
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
    #N   = palc_hyperplane_constraint(δu, δu0, δλ, δλ0, θ, ds)
    N   = pnorm(δu, δu0, δλ, δλ0, ds, alg.norm)

    # Set function and return
    F[1:n] .= cache.Ffun
    F[end]  = N
    return nothing
end

function palc_correction_jacobian!(J, uλ, p)
    # Get parameters
    fun     = p[1]
    cache   = p[2]
    alg     = p[3]
    δu0     = cache.δu0
    δλ0     = cache.δλ0
    n       = length(uλ) - 1

    # Evaluate the jacobian and set
    eval_J!(cache.Jfun, cache.Ffun, uλ, fun)
    J[1:n, :] .= cache.Jfun

    # Evaluate the hyperplane constraint jacobian
    #palc_hyperplane_constraint_jacobian!(view(J, n+1, :), δu0, δλ0, θ)
    dpnorm_du!(view(J, n+1, 1:n), δu0, δu0, δλ0, δλ0, alg.norm)
    J[n+1, n+1] = dpnorm_dλ(δu0, δu0, δλ0, δλ0, alg.norm)

    return nothing
end