abstract type AbstractContinuationCallback end
abstract type RootSolveContinuationCallback <: AbstractContinuationCallback end
abstract type InternalRootSolveContinuationCallback <: RootSolveContinuationCallback end
abstract type UserRootSolveContinuationCallback <: RootSolveContinuationCallback end

# This callback allows for performing continuation until a user provided callback function equals zero,
# at which point, the method will find the zero precisely before terminating
"""
    TerminateContinuationCallback

Terminate the continuation when a user-provided callback equals zero.

When a sign change in the user-provided function is detected, a regula-falsi solver finds the precise point
where the callback is satisfied, then terminates.
"""
mutable struct TerminateContinuationCallback{FType} <: UserRootSolveContinuationCallback
    # The callback function
    f::FType # Takes the current iterate as arguments and returns Float64

    # Current and previous callback value
    val_0::Float64

    # Tolerance
    tol::Float64

    @doc"""
        TerminateContinuationCallback(f::F; tol=1e-12)
    
    Constructor for `TerminateContinuationCallback`.
    
    # Arguments
    - `f::Function`: user defined callabck function of form f(u,λ)

    # Kwargs
    - `tol::Float`: Tolerance for regula-falsi solver. Defaults to 1e-12.

    # Examples
    ```julia
    cb_fun = (u, λ) -> u[1] # terminate when first unknown is zero
    cb = TerminateContinuationCallback(cb_fun)
    ```
    """
    function TerminateContinuationCallback(f::F; tol=1e-12) where {F<:Function}
        fwrap = FunctionWrappersWrapper(f, (Tuple{Vector{Float64},Float64},), (Float64,))
        return new{typeof(fwrap)}(fwrap, NaN, tol)
    end
end

# This callback is similar to the TerminateContinuationCallback, but rather than use a
# user provided function, it instead employs an internal mechanism that requires more
# information about the internal state of the continuation process than we want to
# expose to users.
#
# This is what is actual employed internally with a FoldBifurcationTerminationCallback
mutable struct InternalTerminateContinuationCallback{FType} <:
               InternalRootSolveContinuationCallback
    # The callback function
    f::FType # Takes the current iterate as arguments and returns Float64

    # Current and previous callback value
    val_0::Float64

    # Tolerance
    tol::Float64

    # Constructer
    function InternalTerminateContinuationCallback(
        f::F, cache, alg, prob; tol=1e-12
    ) where {F<:Function}
        fwrap = FunctionWrappersWrapper(
            f,
            (Tuple{Vector{Float64},Float64,typeof(cache),typeof(alg),typeof(prob)},),
            (Float64,),
        )

        return new{typeof(fwrap)}(fwrap, NaN, tol)
    end
end

# This callback is functionally identical to the InternalTerminateContinuationCallback, in terms of the fact that
# it is an internal-only callback that has root-solve functionality, but it callbacks of this type will not
# terminate the calculated point, but rather save it to the cache. This will let us save folds without restarting the algorithm, for instance.
mutable struct InternalDetectionCallback{FType} <: InternalRootSolveContinuationCallback
    # The callback function
    f::FType # Takes the current iterate as arguments and returns Float64

    # Current and previous callback value
    val_0::Float64

    # Tolerance
    tol::Float64

    # Cache for the iterate to use within the root-find, so the true iterate isn't overwritten
    uλ::Vector{Float64}

    point_type::Symbol # symbol to denote type of detected point (e.g., :Fold for fold bifurcations)

    # Constructer
    function InternalDetectionCallback(
        f::F, cache, alg, prob, point_type; tol=1e-12
    ) where {F<:Function}
        fwrap = FunctionWrappersWrapper(
            f,
            (Tuple{Vector{Float64},Float64,typeof(cache),typeof(alg),typeof(prob)},),
            (Float64,),
        )
        return new{typeof(fwrap)}(fwrap, NaN, tol, similar(cache.uλ0), point_type)
    end
end

# Callback sets
mutable struct TerminateContinuationCallbackSet{T<:Tuple} <: RootSolveContinuationCallback
    callbacks::T # Each argument should be a RootSolveContinuationCallback, or else errors will occur later
    # the user CAN costruct with a tuple of anything, but should use the below constructor with varargs
end

# don't need to do any handling like in SciMLBase, since we only have 1 type of callback for now
function TerminateContinuationCallbackSet(callbacks::Union{RootSolveContinuationCallback, Nothing}...)
    TerminateContinuationCallbackSet(callbacks)
end

# Simple continuation callback for analyzing the status of the continuation process.
# Has no zero finding functionality
struct AnalysisContinuationCallback{FType} <: AbstractContinuationCallback
    # The callback function
    f::FType # Takes the continuation cache as single argument and returns Float64

    # Constructer
    function AnalysisContinuationCallback(f::F) where {F<:Function}
        fwrap = FunctionWrappersWrapper(f, (Tuple{PALCCache},), (Nothing,))
        return new{typeof(fwrap)}(fwrap)
    end
end

# Callback initialization
initialize!(cb::Nothing, cache::PALCCache, alg, prob) = nothing
function initialize!(cb::UserRootSolveContinuationCallback, cache::PALCCache, alg, prob)
    cb.val_0 = cb.f(cache.u0, cache.λ0)
    return nothing
end
function initialize!(cb::InternalRootSolveContinuationCallback, cache::PALCCache, alg, prob)
    cb.val_0 = cb.f(cache.u0, cache.λ0, cache, alg, prob)
    return nothing
end
function initialize!(cb::TerminateContinuationCallbackSet, cache::PALCCache, alg, prob)
    # initialize each callback in the set
    @inbounds for i in eachindex(cb.callbacks)
        initialize!(cb.callbacks[i], cache, alg, prob)
    end
    return nothing
end

# Callback update
update!(cb::Nothing, cache::PALCCache, alg, prob) = nothing
function update!(cb::UserRootSolveContinuationCallback, cache::PALCCache, alg, prob)
    cb.val_0 = cb.f(cache.u0, cache.λ0)
    return nothing
end
function update!(cb::InternalRootSolveContinuationCallback, cache::PALCCache, alg, prob)
    cb.val_0 = cb.f(cache.u0, cache.λ0, cache, alg, prob)
    return nothing
end
function update!(cb::TerminateContinuationCallbackSet, cache::PALCCache, alg, prob)
    @inbounds for i in eachindex(cb.callbacks)
        update!(cb.callbacks[i], cache, alg, prob)
    end
end

# Check the root solve callback (returns true if we stepped over zero)
check(cb::Nothing, uλ0, cache::PALCCache, alg, prob) = false
function check(cb::UserRootSolveContinuationCallback, uλ0, cache::PALCCache, alg, prob)
    # Evaluate callback function
    val_1 = call!(cb, uλ0, cache, alg, prob)
    return val_1 * cb.val_0 < 0
end
function check(cb::InternalRootSolveContinuationCallback, uλ0, cache::PALCCache, alg, prob)
    # Evaluate callback function
    val_1 = call!(cb, uλ0, cache, alg, prob)
    return val_1 * cb.val_0 < 0
end

# Call the callback callback
call!(cb::Nothing, cache::PALCCache) = nothing
function call!(cb::AnalysisContinuationCallback, cache::PALCCache)
    cb.f(cache)
    return nothing
end
function call!(cb::UserRootSolveContinuationCallback, uλ0, cache::PALCCache, alg, prob)
    n = length(uλ0) - 1
    u = cache.u_t
    u .= view(uλ0, 1:n)
    return cb.f(u, uλ0[end])
end
function call!(cb::UserRootSolveContinuationCallback, u0, λ0, cache::PALCCache, alg, prob)
    return cb.f(u0, λ0)
end
function call!(cb::InternalRootSolveContinuationCallback, uλ0, cache::PALCCache, alg, prob)
    n = length(uλ0) - 1
    u = cache.u_t
    u .= view(uλ0, 1:n)
    return cb.f(u, uλ0[end], cache, alg, prob)
end
function call!(
    cb::InternalRootSolveContinuationCallback, u0, λ0, cache::PALCCache, alg, prob
)
    return cb.f(u0, λ0, cache, alg, prob)
end

# Functions for handling different types of termination callbacks
function handle_termination_callback(cb, cache, alg, p)
    return cb
end
function handle_termination_callback(cb::TerminateContinuationCallbackSet{T}, cache, alg, p) where {T<:Tuple}
    handle_cb_map_fun = cb-> handle_termination_callback(cb, cache, alg, p)
    cb = TerminateContinuationCallbackSet(map(handle_cb_map_fun, cb.callbacks))
    return cb
end

# functions for handling general detection callbacks
function handle_detection_callback(cb, cache, alg, p)
    return cb
end

function perform_detection_callback!(cache, alg, prob, solvers, callback::Nothing, uλ, λmax, λmin, trace)
    return true
end

function perform_detection_callback!(cache, alg, prob, solvers, callback::InternalDetectionCallback, uλc, λmax, λmin, trace)
    # If the success is false, the step-size will be halved and step will be retried
    det_success = false
    
    # store original value of iterate
    callback.uλ .= uλc

    # Check callback
    cb_trig = check(callback, uλc, cache, alg, prob)

    if cb_trig
        uλc .= view(uλc, :)

        rf_succ = palc_target_callback_event!(
                    uλc, cache, alg, prob, solvers, callback, trace
        )
        
        if rf_succ
            # only save detected point if it is within bounds
            if uλc[end] <= λmax && uλc[end] >= λmin
                # Save the detected point to the cache
                push!(cache.detected_points, (copy(uλc[1:end-1]), uλc[end], callback.point_type))

                # Also push the point to the curve (but don't set a successful iterate)
                push!(cache.br, (copy(uλc[1:end-1]), uλc[end]))
                
            end
            # If the root-find was successful, we consider the detection a success
            # regardless of if we saved the point or not, since it may have been out of bounds
            det_success = true
        end

    else
        det_success = true # if the callback is not triggered, consider it a success (i.e., not a failure)
    end

    # reset iterate to original value
    uλc .= callback.uλ

    return det_success

end