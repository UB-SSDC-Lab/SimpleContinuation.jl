abstract type AbstractStepLimiter end

"""
    struct CorrectionStepLimiter

A type to limit the step size in the correction step.

Restricts the Euclidean distance between the prediction and correction to be less than some `frac` of `Δs` at the current iteration.
If the distance is larger, the correction step is rejected despite success of the Newton-Raphson solver.

# Fields
- `frac::Float64`: The fraction of the current PALC step size `Δs` that the correction step length should be less than.
"""
struct CorrectionStepLimiter <: AbstractStepLimiter

    frac::Float64

    @doc"""
        function CorrectionStepLimiter(; frac=0.1)

    Constructor for the correction step limiter. 
    
    # Kwargs
    - `frac::Float64`: The fraction of the current PALC step size `Δs` that the correction step length should be less than. Defaults to 0.1.
    """
    function CorrectionStepLimiter(; frac=0.1)
        return new(frac)
    end

end

function enforce_local_correction(correction_success, uλ, uλpred, ds, alg, sl::Nothing)
    # If no step limiter, allow step to proceed
    return false
end

function enforce_local_correction(correction_success, uλ, uλpred, ds, alg, sl::AbstractStepLimiter)
    error("Unrecognized correction step limiter!")
    return false
end

function enforce_local_correction(correction_success, uλ, uλpred, ds, alg, sl::CorrectionStepLimiter)
    reject_step=false
    if correction_success # only want to do this check if the Newton solve was actually successful
        n = length(uλ)-1
        δn = sqrt(alg.inner_prod(view(uλ,1:n)-view(uλpred, 1:n), uλ[end]-uλpred[end])) # norm of change between pred and corr
        reject_step = δn / abs(ds) > sl.frac ? true : false
    end
    return reject_step
end


function palc_correction!(
    cache,
    alg,
    p::ContinuationProblem,
    solvers,
    dsmin,
    dsmax,
    term_callback,
    analysis_callback,
    detection_callback,
    step_limiter,
    trace,
)
    # Get cache variables
    u0 = cache.u0
    λ0 = cache.λ0
    δu0 = cache.δu0
    δλ0 = cache.δλ0
    uλpred = cache.uλpred
    n = length(δu0)

    # Compute inner product of tangent with itself
    dotδ = alg.inner_prod(δu0, δλ0)

    # Get problem variables
    λmin = p.λ_bounds[1]
    λmax = p.λ_bounds[2]

    # Solve nonlinear problem (reducing step-size if necessary)
    attempts = 0
    success = true
    done = false
    hit_bnd = NaN
    cb_trig = false
    rf_succ = false
    while !done
        # Update attempts
        attempts += 1

        # Compute α for ds
        α = cache.ds / dotδ

        # Update uλpred (clamping ds to try and stay in λ bounds)
        # If clamped, set hit_bnd to the bound hit and we'll resolve
        # if successful with constant λ

        uλpred[end] = λ0 + α * δλ0
        if uλpred[end] < λmin
            α = (λmin - λ0) / δλ0
            cache.ds = α * dotδ
            uλpred[end] = λmin
            hit_bnd = λmin
            print_correction_trace(cache, trace, 2)
        elseif uλpred[end] > λmax
            α = (λmax - λ0) / δλ0
            cache.ds = α * dotδ
            uλpred[end] = λmax
            hit_bnd = λmax
            print_correction_trace(cache, trace, 2)
        else
            print_correction_trace(cache, trace, 1)
        end

        uλpred[1:n] .= u0 .+ α .* δu0

        # Solve the palc nonlinear problem
        uλ, retcode = solve_palc_nlp!(solvers, uλpred, trace)

        # Check if successful
        correction_success = SciMLBase.successful_retcode(retcode) # NL solve was successful

        # check if this step should be rejected anyway
        reject_step = enforce_local_correction(correction_success, uλ, uλpred, cache.ds, alg, step_limiter)
        

        if correction_success && !reject_step

            # Check if callback triggered
            cb_trig = check(term_callback, uλ, cache, alg, p)

            # If callback triggered, perform regula falsi root finding method and update uλ
            if cb_trig
                rf_succ = palc_target_callback_event!(
                    uλ, cache, alg, p, solvers, term_callback, trace
                )
                hit_bnd = NaN # Reset since we're likely not stepping as far and will recheck
            end

            # Handle detection callback
            # Returns true if: not triggered (or is nothing), triggered and rootfind successful
            # the return is only used to reduce the step-size if failure occurs
            # all saving is internal to the perform function, only saves if computed point is within bounds
            # this does NOT edit the iterate uλ (well, technically it does, but it puts it resets it after)
            # I think this leaves the possibility of a failed termination RF + successful RF here creating duplicate points, 
            # so consider adding additional checks
            detection_success = perform_detection_callback!(cache, alg, p, solvers, detection_callback, uλ, λmax, λmin, trace)

            # Check if we crossed the boundary
            if uλ[end] < λmin
                hit_bnd = λmin
            elseif uλ[end] > λmax
                hit_bnd = λmax
            end

            # Determine if done
            if (cb_trig && !rf_succ) || !detection_success # Triggered callback but rootfind was unsuccessful OR detection triggered and was unsuccessful
                cb_trig = false
                hit_bnd = NaN
                if abs(cache.ds)==dsmin # check if ds=dsmin to avoid getting stuck repeating the failed rootfind
                    done = true
                    success = false
                    set_min_stepsize_retcode!(cache) # failed termination if this occurs
                end
                scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
            elseif isnan(hit_bnd) # bound was not hit (regardless of term callback)
                # Push solution and set done
                set_successful_iterate!(cache, uλ)
                done = true

                # Print trace if desired
                print_correction_trace(cache, trace, 3)
            else

                # Target solution on boundary
                flag = palc_target_solution_on_boundary!(cache, hit_bnd, solvers, trace)

                # If targeting solution on boundary was successful, we're done. Otherwise, reduce ds
                if flag
                    # Only set successful if targeting worked
                    # Update cache without pushing solution to curve
                    set_successful_iterate!(cache, uλ, false)

                    # Print trace if desired
                    print_correction_trace(cache, trace, 3)
                    done = true
                else
                    hit_bnd = NaN
                    scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
                end
            end
        else # solve is not successful or step rejected, so reduce ds
            if abs(cache.ds) == dsmin
                done = true
                success = false
                set_min_stepsize_retcode!(cache) # update ret with 'done' condition. This won't be overwritten since success=false (see continuation.jl) 
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
    success && update!(term_callback, cache, alg, p)
    success && update!(detection_callback, cache, alg, p)

    # Call the analysis callback if we were successful
    success && call!(analysis_callback, cache)

    # Handle termination flag
    terminate_continuation = !isnan(hit_bnd) || cb_trig # hit bound or triggered callback

    if terminate_continuation
        set_successful_retcode!(cache, hit_bnd, cb_trig)
    end

    return success, terminate_continuation
end

## Correction method if we are using a set of callbacks
# Only difference here is that instead of looking for 1 callback being triggered, we look for if any of the set is triggered sequentially, then perform RF.
# Note: should probably later improve this to cover the case that multiple callbacks trigger in a single step, but for now we are looking at them sequentially
function palc_correction!(
    cache,
    alg,
    p::ContinuationProblem,
    solvers,
    dsmin,
    dsmax,
    term_callback::TerminateContinuationCallbackSet,
    analysis_callback,
    detection_callback,
    step_limiter,
    trace,
)
    # Get cache variables
    u0 = cache.u0
    λ0 = cache.λ0
    δu0 = cache.δu0
    δλ0 = cache.δλ0
    uλpred = cache.uλpred
    n = length(δu0)

    # Compute inner product of tangent with itself
    dotδ = alg.inner_prod(δu0, δλ0)

    # Get problem variables
    λmin = p.λ_bounds[1]
    λmax = p.λ_bounds[2]

    # Solve nonlinear problem (reducing step-size if necessary)
    attempts = 0
    success = true
    done = false
    hit_bnd = NaN
    cb_trig = false
    rf_succ = false
    triggered_callback = nothing # which (if any) callback was triggered
    while !done
        # Update attempts
        attempts += 1

        # Compute α for ds
        α = cache.ds / dotδ

        # Update uλpred (clamping ds to try and stay in λ bounds)
        # If clamped, set hit_bnd to the bound hit and we'll resolve
        # if successful with constant λ

        uλpred[end] = λ0 + α * δλ0
        if uλpred[end] < λmin
            α = (λmin - λ0) / δλ0
            cache.ds = α * dotδ
            uλpred[end] = λmin
            hit_bnd = λmin
            print_correction_trace(cache, trace, 2)
        elseif uλpred[end] > λmax
            α = (λmax - λ0) / δλ0
            cache.ds = α * dotδ
            uλpred[end] = λmax
            hit_bnd = λmax
            print_correction_trace(cache, trace, 2)
        else
            print_correction_trace(cache, trace, 1)
        end

        uλpred[1:n] .= u0 .+ α .* δu0

        # Solve the palc nonlinear problem
        uλ, retcode = solve_palc_nlp!(solvers, uλpred, trace)

        # Check if successful
        correction_success = SciMLBase.successful_retcode(retcode) # NL solve was successful

        # check if this step should be rejected anyway
        reject_step = enforce_local_correction(correction_success, uλ, uλpred, cache.ds, alg, step_limiter)

        if correction_success && !reject_step

            # Check if callback triggered (iterating through in order)
            for jj in eachindex(term_callback.callbacks)
                cb_trig = check(term_callback.callbacks[jj], uλ, cache, alg, p)
                if cb_trig # break once one is triggered
                    triggered_callback = jj
                    break
                end
            end

            # If callback triggered, perform regula falsi root finding method and update uλ
            if cb_trig
                rf_succ = palc_target_callback_event!(
                    uλ, cache, alg, p, solvers, term_callback.callbacks[triggered_callback], trace
                )
                hit_bnd = NaN # Reset since we're likely not stepping as far and will recheck
            end

            detection_success = perform_detection_callback!(cache, alg, p, solvers, detection_callback, uλ, λmax, λmin, trace)

            # Check if we crossed the boundary
            if uλ[end] < λmin
                hit_bnd = λmin
            elseif uλ[end] > λmax
                hit_bnd = λmax
            end

            if cb_trig && !rf_succ || !detection_success || reject_step # Triggered callback but rootfind was unsuccessful
                cb_trig = false
                hit_bnd = NaN
                if abs(cache.ds)==dsmin # check if ds=dsmin to avoid getting stuck repeating the failed rootfind
                    done = true
                    success = false
                    set_min_stepsize_retcode!(cache) # failed termination if this occurs
                end
                scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
            elseif isnan(hit_bnd)
                # Push solution and set done
                set_successful_iterate!(cache, uλ)
                done = true

                # Print trace if desired
                print_correction_trace(cache, trace, 3)
            else

                # Target solution on boundary
                flag = palc_target_solution_on_boundary!(cache, hit_bnd, solvers, trace)

                # If targeting solution on boundary was successful, we're done. Otherwise, reduce ds
                if flag
                    #  Update cache without pushing solution to curve
                    set_successful_iterate!(cache, uλ, false)

                    # Print trace if desired
                    print_correction_trace(cache, trace, 3)
                    done = true
                else
                    hit_bnd = NaN
                    scale_and_clamp_ds!(cache, 0.5, dsmin, dsmax)
                end
            end
        else # solve is not successful or step rejected
            if abs(cache.ds) == dsmin
                done = true
                success = false
                set_min_stepsize_retcode!(cache) # update ret with 'done' condition. This won't be overwritten since success=false (see continuation.jl) 
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
    success && update!(term_callback, cache, alg, p)
    success && update!(detection_callback, cache, alg, p)

    # Call the analysis callback if we were successful
    success && call!(analysis_callback, cache)

    # Handle termination flag
    terminate_continuation = !isnan(hit_bnd) || cb_trig # hit bound or triggered callback

    if terminate_continuation
        set_successful_retcode!(cache, hit_bnd, cb_trig, triggered_callback)
    end

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
            δu = cache.u_0
            δu .= cache.br[end][1] .- cache.br[end - 1][1]
            δλ = cache.br[end][2] - cache.br[end - 1][2]
            if cache.ds < 0.0
                δu .*= -1.0
                δλ = -δλ
            end
            dp = dot(δu, cache.δu0) + δλ * cache.δλ0
            r = dp / (sqrt(dot(δu, δu) + δλ^2) * norm(cache.δuλ0))
            θ = acosd(clamp(r, -1.0, 1.0))
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
    abs_ds = abs(cache.ds)
    new_ds = clamp(scale * abs_ds, dsmin, dsmax)
    cache.ds = sign_ds * new_ds
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

# Function to find when callback = 0 with regula falsi method and natural continuation
function nc_target_callback_event!(uλ, cache, alg, prob, solvers, callback, trace)
    # Get callback value at boundaries
    f_0 = callback.val_0
    f_1 = call!(callback, uλ, cache, alg, prob)

    # Get parameter values at boundaries
    λ_0 = cache.λ0
    λ_1 = uλ[end]

    # Get inputs at boundaries
    u_0 = cache.u_0
    u_1 = cache.u_1
    u_t = cache.u_t
    n = length(uλ) - 1
    u_0 .= cache.u0
    u_1 .= view(uλ, 1:n)

    # Begin loop
    done = false
    success = false
    while !done
        λ_2 = λ_0 - f_0 * (λ_1 - λ_0) / (f_1 - f_0)
        u_t .= u_0 .+ ((λ_2 - λ_0) / (λ_1 - λ_0)) .* (u_1 .- u_0)

        set_natural_continuation_parameter!(cache, λ_2)
        usol, retcode = solve_natural_nlp!(solvers, u_t, trace)

        success_flag = SciMLBase.successful_retcode(retcode)
        if success_flag
            # Call the callback function
            f_2 = call!(callback, usol, λ_2, cache, alg, prob)

            if abs(f_2) <= callback.tol || abs(λ_1 - λ_0) < callback.tol
                # Set flags
                done = true
                success = true

                # Update iterate
                uλ[1:n] .= u_0
                uλ[end] = λ_0
            elseif f_0 * f_2 < 0
                u_1 .= usol
                λ_1 = λ_2
                f_1 = f_2
            else
                u_0 .= usol
                λ_0 = λ_2
                f_0 = f_2
            end
        else
            done = true
        end
    end

    return success
end

# Function to find when callback = 0 with regula falsi method and PALC
function palc_target_callback_event!(uλ, cache, alg, prob, solvers, callback, trace)
    # Get callback value at boundaries
    f_0 = callback.val_0
    f_1 = call!(callback, uλ, cache, alg, prob)

    # Get delta-arclength values at boundaries
    dotδ = alg.inner_prod(cache.δu0, cache.δλ0)
    start_ds = cache.ds
    ds_0 = 0.0
    ds_1 = start_ds

    # Get inputs at boundaries
    u_0 = cache.u_0
    u_1 = cache.u_1
    u_t = cache.u_t
    n = length(uλ) - 1

    u_0 .= cache.u0
    λ_0 = cache.λ0
    u_1 .= view(uλ, 1:n)
    λ_1 = uλ[end]

    # Begin loop
    done = false
    success = false
    while !done
        ds_t = ds_0 - f_0 * (ds_1 - ds_0) / (f_1 - f_0)
        α_t = ds_t / dotδ
        u_t .= cache.u0 .+ α_t .* cache.δu0
        λ_t = cache.λ0 + α_t * cache.δλ0

        cache.ds = ds_t
        cache.uλpred[1:n] .= u_t
        cache.uλpred[end] = λ_t
        uλ_t, retcode = solve_palc_nlp!(solvers, cache.uλpred, trace)

        success_flag = SciMLBase.successful_retcode(retcode)
        if success_flag
            # Call the callback function
            f_t = call!(callback, uλ_t, cache, alg, prob)

            if abs(ds_1 - ds_0) < callback.tol
                # Set flags
                done = true
                success = true

                # Update iterate
                uλ .= uλ_t
            elseif f_0 * f_t < 0
                ds_1 = ds_t
                u_1 .= view(uλ_t, 1:n)
                λ_1 = uλ_t[end]
                f_1 = f_t
            else
                ds_0 = ds_t
                u_0 .= view(uλ_t, 1:n)
                λ_0 = uλ_t[end]
                f_0 = f_t
            end
        else
            done = true
        end
    end

    # Reset ds to original value
    cache.ds = start_ds

    return success
end

# ===== Nonlinear solve functions
function palc_correction_function!(F, uλ, p)
    # Get parameters
    fun = p[1]
    cache = p[2]
    alg = p[3]
    δu0 = cache.δu0
    δλ0 = cache.δλ0
    ds = cache.ds

    # Get u and λ
    n = length(uλ) - 1
    u = view(uλ, 1:n)
    λ = uλ[end]

    # Evaluate the hyperplane constraint
    # We'll use F to store the differences here so we can
    # support ForwardDiff evaluations
    δu = view(F, 1:n)
    δu .= u .- cache.u0
    δλ = λ - cache.λ0
    N = palc_norm(δu, δu0, δλ, δλ0, ds, alg.inner_prod)

    # Set the hyperplane constraint
    F[end] = N

    # Evaluate the function
    eval_f!(view(F, 1:n), uλ, fun)

    return nothing
end

function palc_correction_jacobian!(J, uλ, p)
    # Get parameters
    fun = p[1]
    cache = p[2]
    alg = p[3]
    δu0 = cache.δu0
    δλ0 = cache.δλ0
    n = length(uλ) - 1

    # Evaluate the jacobian and set
    eval_J!(cache.Jfun, cache.Ffun, uλ, fun)
    J[1:n, :] .= cache.Jfun

    # Evaluate the hyperplane constraint jacobian
    palc_norm_dδu!(view(J, n + 1, 1:n), δu0, alg.inner_prod)
    J[n + 1, n + 1] = palc_norm_dδλ(δλ0, alg.inner_prod)

    return nothing
end

function set_successful_retcode!(cache, hit_bnd, cb_trig)
    # recover success condition and set
    if isnan(hit_bnd) && cb_trig
        cache.ret = :Callback
    elseif !cb_trig
        cache.ret = :HitBound
    end
end

function set_successful_retcode!(cache, hit_bnd, cb_trig, jj)
    # recover success condition and set
    if isnan(hit_bnd) && cb_trig
        retstr = "Callback$jj"
        cache.ret = Symbol(retstr)
    elseif !cb_trig
        cache.ret = :HitBound
    end
end

function set_min_stepsize_retcode!(cache)
    cache.ret = :MinimumStepSize
    return nothing
end