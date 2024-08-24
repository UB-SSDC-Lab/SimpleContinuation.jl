
# ===== Bordered Predictor
function palc_prediction!(cache::PALCCache, alg::PALC{Bordered}, p::ContinuationProblem, solvers, trace)
    # Fill bordered matrix
    set_boardered_matrix!(cache, alg, p)

    # Update lp cache with new matrix
    set_lp_matrix!(solvers, cache.bordered_mat)

    # Solve the linear problem
    x = solve_lp!(solvers)

    # Set the new tangent, making sure we keep moving in the same direction
    scale_predicted_tangent!(x, cache)
    update_tangent!(cache, x)

    # Print trace if desired
    print_prediction_trace(cache, trace)

    return nothing
end

function scale_predicted_tangent!(x, cache::PALCCache)
    α  = 1.0 / norm(x)
    α *= sign(dot(x, cache.δuλ0))
    x *= α
    return nothing
end

function set_boardered_matrix!(cache::PALCCache, alg::PALC{Bordered}, p::ContinuationProblem)
    # Get parameters
    fun     = p.f
    uλ0     = cache.uλ0
    F       = cache.Ffun
    J       = cache.Jfun
    A       = cache.bordered_mat

    # Evaluate the function jacobian and set in bm
    n = length(uλ0) - 1
    eval_J!(J, F, uλ0, fun)
    A[1:n, 1:n+1] .= J

    # Remaining
    A[n+1, :] .= cache.δuλ0
    #dpnorm_du!(view(A, n+1, 1:n), cache.δu0, cache.δu0, cache.δλ0, cache.δλ0, alg.norm)
    #A[n+1, n+1] = dpnorm_dλ(cache.δu0, cache.δu0, cache.δλ0, cache.δλ0, alg.norm)

    return nothing
end
function set_boardered_matrix!(
    cache::PALCCache, p::ContinuationProblem{FT},
) where {FT <: SparseContinuationFunction}
    # Get parameters
    fun     = p.f
    uλ0     = cache.uλ0
    F       = cache.Ffun
    J       = cache.Jfun
    A       = cache.bordered_mat

    # Evaluate the function jacobian and set in bm
    n = length(uλ0) - 1
    eval_J!(J, F, uλ0, fun)

    rows = rowvals(J)
    vals = nonzeros(J)
    nr, nc  = size(J)
    for j = 1:nc
        for i in nzrange(J,j)
            row = rows[i]
            A[row,j] = vals[i]
        end
        A[n+1, j] = cache.δuλ0[i]
    end

    #dpnorm_du!(view(A, n+1, 1:n), cache.δu0, cache.δu0, cache.δλ0, cache.δλ0, alg.norm)
    #A[n+1, n+1] = dpnorm_dλ(cache.δu0, cache.δu0, cache.δλ0, cache.δλ0, alg.norm)

    return nothing
end

# Just doing nothing for now in all caes as I'm not really sure if there's any relevant 
# information to print here (we're already going to print the predicted update to λ
# in the correction step)
function print_prediction_trace(cache::PALCCache, trace::Silent)
    return nothing
end
function print_prediction_trace(cache::PALCCache, trace::NonSilentTraceLevel)
    println("Computed boardered prediction.")
    return nothing
end

# ===== Secant Predictor
function palc_prediction!(cache::PALCCache, alg::PALC{Secant}, p::ContinuationProblem, solvers, trace)
    # Set tangent direction with secant prediction
    if length(cache.br) < 2
        # Just use initial tangent so do nothing here...
        return nothing
    else
        # Set directions
        cache.δu0 .= cache.br[end][1] .- cache.br[end-1][1]
        cache.δλ0  = cache.br[end][2]  - cache.br[end-1][2]

        # Set full direction
        n = length(cache.δu0)
        cache.δuλ0[1:n] .= cache.δu0
        cache.δuλ0[n+1]  = cache.δλ0

        # Scale
        nδuλ0 = norm(cache.δuλ0)
        cache.δuλ0 ./= nδuλ0
        cache.δu0   .= view(cache.δuλ0, 1:n)
        cache.δλ0    = cache.δuλ0[n+1]

        return nothing
    end
end