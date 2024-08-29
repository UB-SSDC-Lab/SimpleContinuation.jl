abstract type AbstractContinuationCallback end
abstract type RootSolveContinuationCallback <: AbstractContinuationCallback end

# This callback allows for performing continuation until the callback function equals zero,
# at which point, the method will find the zero precisly before terminating
mutable struct TerminateContinuationCallback{FType} <: RootSolveContinuationCallback
    # The callback function
    f::FType # Takes the current iterate as arguments and returns Float64

    # Current and previous callback value
    val_0::Float64

    # Tolerance
    tol::Float64

    # Constructer
    function TerminateContinuationCallback(f::F; tol = 1e-12) where {F <: Function}
        fwrap = FunctionWrappersWrapper(
            f, (Tuple{Vector{Float64}, Float64, PALCCache},), (Float64,),
        )
        new{typeof(fwrap)}(fwrap, NaN, tol)
    end
end

# The following callback will likely not work great until we implement a way to stop checking callback for a little bit
# after finding a zero (otherwise, well sometimes continue to detect the same root)
# # This callback will interate precisly to the point where the callback function equals zero,
# # but will not terminate the continuation process
# mutable struct IterateToContinuationCallback{FType} <: RootSolveContinuationCallback
#     # The callback function
#     f::FType # Takes the current iterate as arguments and returns Float64

#     # Current and previous callback value
#     val_0::Float64
#     val_1::Float64

#     # Constructer
#     function IterateToContinuationCallback(f::F; tol = 1e-12) where {F <: Function}
#         fwrap = FunctionWrappersWrapper(f, (Vector{Float64}, Float64), (Float64,))
#         new{typeof(fwrap)}(fwrap, NaN, NaN, tol)
#     end
# end

# Simple continuation callback for analyzing the status of the continuation process.
# Has no zero finding functionality
struct AnalysisContinuationCallback{FType} <: AbstractContinuationCallback
    # The callback function
    f::FType # Takes the continuation cache as single argument and returns Float64

    # Constructer
    function AnalysisContinuationCallback(f::F) where {F <: Function}
        fwrap = FunctionWrappersWrapper(f, (Tuple{PALCCache},), (Nothing,))
        new{typeof(fwrap)}(fwrap)
    end
end

# Callback initialization
initialize!(cb::Nothing, cache::PALCCache) = nothing
function initialize!(cb::RootSolveContinuationCallback, cache::PALCCache)
    cb.val_0 = cb.f(cache.u0, cache.λ0, cache)
    return nothing
end

# Callback update
update!(cb::Nothing, cache::PALCCache) = nothing
function update!(cb::RootSolveContinuationCallback, cache::PALCCache)
    cb.val_0 = cb.f(cache.u0, cache.λ0, cache)
    return nothing
end

# Check the root solve callback (returns true if we stepped over zero)
check(cb::Nothing, uλ0, cache::PALCCache) = false
function check(cb::RootSolveContinuationCallback, uλ0, cache::PALCCache)
    # Evaluate callback function
    val_1 = call!(cb, uλ0, cache)
    return val_1*cb.val_0 < 0
end

# Call the callback callback
call!(cb::Nothing, cache::PALCCache) = nothing
function call!(cb::AnalysisContinuationCallback, cache::PALCCache)
    cb.f(cache)
    return nothing
end
function call!(cb::RootSolveContinuationCallback, uλ0, cache::PALCCache)
    n = length(uλ0) - 1
    u = cache.u_t; u .= view(uλ0, 1:n)
    return cb.f(u, uλ0[end], cache)
end
function call!(cb::RootSolveContinuationCallback, u0, λ0, cache::PALCCache)
    return cb.f(u0, λ0, cache)
end