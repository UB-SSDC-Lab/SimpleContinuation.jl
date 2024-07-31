
struct ContinuationProblem{F}
    # The ContinuationFunction
    f::F

    # Initial solution
    u0::Vector{Float64}
    λ0::Float64

    # Continuation parameter bounds
    λ_bounds::Tuple{Float64,Float64}

    # Constructor
    function ContinuationProblem(
        f::F,
        u0::Vector{Float64},
        λ0::Float64,
        λ_bounds::Tuple{Float64,Float64},
    ) where F
        new{F}(f, u0, λ0, λ_bounds)
    end
end

