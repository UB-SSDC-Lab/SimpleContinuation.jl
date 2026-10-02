"""
    ContinuationProblem{F}

Contains all problem information.

# Fields:
- `f::F`: A `ContinuationFunction` instance for the problem to solve.
- `u0::Vector{Float64}`: Initial value for unknowns.
- `λ0::Float64`: Initial value for continuation parameter.
- `λ_bounds::Tuple{Float64, Float64}`: Defines acceptable range of  `λ`, in format `(λ_min, λ_max)`
"""
struct ContinuationProblem{F}
    # The ContinuationFunction
    f::F

    # Initial solution
    u0::Vector{Float64}
    λ0::Float64

    # Continuation parameter bounds
    λ_bounds::Tuple{Float64,Float64}

    # Constructor
    @doc"""
        ContinuationProblem(f, u0, λ0, λ_bounds)
    
    Constructor for `ContinuationProblem`.

    # Arguments:
    - `f::F`: A `ContinuationFunction` instance for the problem to solve.
    - `u0::Vector{Float64}`: Initial value for unknowns.
    - `λ0::Float64`: Initial value for continuation parameter.
    - `λ_bounds::Tuple{Float64, Float64}`: Defines acceptable range of  `λ`, in format `(λ_min, λ_max)`

    # Returns
    - `prob::ContinuationProblem{F}`: New `ContinuationProblem` instance.
    """
    function ContinuationProblem(
        f::F, u0::Vector{Float64}, λ0::Float64, λ_bounds::Tuple{Float64,Float64}
    ) where {F}
        return new{F}(f, u0, λ0, λ_bounds)
    end
end
