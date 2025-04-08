abstract type AbstractContinuationCallback end
abstract type RootSolveContinuationCallback <: AbstractContinuationCallback end
abstract type InternalRootSolveContinuationCallback <: RootSolveContinuationCallback end
abstract type UserRootSolveContinuationCallback <: RootSolveContinuationCallback end

# This callback allows for performing continuation until a user provided callback function equals zero,
# at which point, the method will find the zero precisely before terminating
mutable struct TerminateContinuationCallback{FType} <: UserRootSolveContinuationCallback
    # The callback function
    f::FType # Takes the current iterate as arguments and returns Float64

    # Current and previous callback value
    val_0::Float64

    # Tolerance
    tol::Float64

    # Constructor
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
