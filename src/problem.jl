"""
    ContinuationProblem{F}

Contains all problem information.

# Fields:
- `f::F`: The in-place continuation function, in the form `f(F, u, λ)=0`
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
        - `f::F`: The (in-place) continuation function, in the form `f(F, u, λ)=0`
        - `u0::Vector{Float64}`: Initial value for unknowns.
        - `λ0::Float64`: Initial value for continuation parameter.
        - `λ_bounds::Tuple{Float64, Float64}`: Defines acceptable range of  `λ`, in format `(λ_min, λ_max)`

    # Returns
        `::ContinuationProblem{F}`: New `ContinuationProblem` instance.

    # Examples
    ```julia
    f = (F,u,λ) -> begin # random function
        F[1] = u[1]^2*log(u[2])*(1-λ)-λ*(u[1]^u[2]/3)
        F[2] = u[2]*λ
        nothing
    end
    u0 = rand(2)
    λ0 = 0.0
    λ_bounds = (0.0,1.0)

    cprob = ContinuationProblem(
        f,
        u0,
        λ0,
        λ_bounds
    )
    ```
    """
    function ContinuationProblem(
        f::F, u0::Vector{Float64}, λ0::Float64, λ_bounds::Tuple{Float64,Float64}
    ) where {F}
        return new{F}(f, u0, λ0, λ_bounds)
    end
end
