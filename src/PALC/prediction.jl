
# ===== Bordered Predictor
function palc_prediction!(cache::PALCCache, alg::PALC{Bordered}, p::ContinuationProblem, solvers, trace)
    # Fill bordered matrix
    set_boardered_matrix!(cache, alg, p)

    # Update lp cache with new matrix
    set_lp_matrix!(solvers, cache.bordered_mat)

    # Solve the linear problem
    x = solve_lp!(solvers)

    # Set the new tangent, making sure we keep moving in the same direction
    scale_predicted_tangent!(x, cache, alg)
    update_tangent!(cache, x)

    # Print trace if desired
    print_prediction_trace(cache, trace)

    return nothing
end

function scale_predicted_tangent!(x, cache::PALCCache, alg::PALC{Bordered})
    n  = length(x) - 1
    xn = sqrt(alg.dot(view(x, 1:n), x[n+1]))
    α  = sign(alg.dot(view(x, 1:n), cache.δu0, x[n+1], cache.δλ0)) / xn
    x .*= α
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
    ddotdu1!(view(A, n+1, 1:n), cache.δu0, alg.dot)
    A[n+1, n+1] = ddotdλ1(cache.δλ0, alg.dot)

    return nothing
end

function set_boardered_matrix!(
    cache::PALCCache, 
    alg::PALC{Bordered}, 
    p::ContinuationProblem{<:SparseContinuationFunction{FT,JuT,JT}},
) where {FT, JuT <: Nothing, JT <: Nothing}
    # Get parameters
    fun     = p.f
    uλ0     = cache.uλ0
    F       = cache.Ffun
    A       = cache.bordered_mat

    # Evaluate the function jacobian with ForwardDiff and set in bm
    n       = length(uλ0) - 1
    dfun    = @closure (du,uλ) -> eval_f!(du, uλ, fun)
    ForwardDiff.jacobian!(view(A, 1:n, 1:n+1), dfun, F, uλ0)

    # Remaining
    ddotdu1!(view(A, n+1, 1:n), cache.δu0, alg.dot)
    A[n+1, n+1] = ddotdλ1(cache.δλ0, alg.dot)

    return nothing
end

function set_boardered_matrix!(
    cache::PALCCache, 
    alg::PALC{Bordered},
    p::ContinuationProblem{FT},
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
    end

    ddotdu1!(view(A, n+1, 1:n), cache.δu0, alg.dot)
    A[n+1, n+1] = ddotdλ1(cache.δλ0, alg.dot)

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
        n = length(cache.δu0)

        # Set directions
        cache.δu0 .= cache.br[end][1] .- cache.br[end-1][1]
        cache.δλ0  = cache.br[end][2]  - cache.br[end-1][2]

        # Compute sign of dot product of secant and current tangent
        sdot = sign(alg.dot(cache.δu0, view(cache.δuλ0, 1:n), cache.δλ0, cache.δuλ0[n+1]))

        # Compute norm of secant
        nδuλ0 = sqrt(alg.dot(cache.δu0, cache.δλ0))

        # Set full direction
        cache.δuλ0[1:n] .= cache.δu0
        cache.δuλ0[n+1]  = cache.δλ0

        # Scale
        α = sdot / nδuλ0
        cache.δuλ0 .*= α
        cache.δu0 .= view(cache.δuλ0, 1:n)
        cache.δλ0 = cache.δuλ0[n+1]

        return nothing
    end
end