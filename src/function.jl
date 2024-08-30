abstract type AbstractContinuationFunction end

# A container for user provided functions. For now, we're just 
# going to handle functions involvng vectors and matrices of Float64s,
# or views thereof. More functionality can be added later as needed.
struct ContinuationFunction{has_full_J,FType,JuType,JλType,JType} <: AbstractContinuationFunction
    f::FType
    Ju::JuType
    Jλ::JλType
    J::JType

    # User provided function and Jacobian wrt u and λ seperately
    function ContinuationFunction{has_full_J}(
        f::Fi, Ju::Jui, Jλ::Jλi,
    ) where {has_full_J <: Val{false}, Fi <: Function, Jui <: Function, Jλi <: Function}
        # Construct input argument types
        λT = Float64
        VT = (
            Vector{Float64},
            SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true},
            SubArray{Float64,1,Matrix{Float64},Tuple{Base.Slice{Base.OneTo{Int}},Int},true},
        )
        MT = (
            Matrix{Float64},
            SubArray{Float64,2,Matrix{Float64},Tuple{Base.Slice{Base.OneTo{Int}},UnitRange{Int}},true},
        )
        fargtypes = (
            Tuple{VT[1],VT[1],λT},
            Tuple{VT[1],VT[2],λT},
        )
        Juargtypes = (
            Tuple{MT[1],VT[1],VT[1],λT},
            Tuple{MT[2],VT[1],VT[1],λT},
            Tuple{MT[1],VT[1],VT[2],λT},
            Tuple{MT[2],VT[1],VT[2],λT},
        )
        Jλargtypes = (
            Tuple{VT[1],VT[1],VT[1],λT},
            Tuple{VT[2],VT[1],VT[1],λT},
            Tuple{VT[3],VT[1],VT[1],λT},
            Tuple{VT[1],VT[1],VT[2],λT},
            Tuple{VT[2],VT[1],VT[2],λT},
            Tuple{VT[3],VT[1],VT[2],λT},
        )

        # Construct function wrappers
        fwrap   = FunctionWrappersWrapper(f, fargtypes, (Nothing,Nothing,))
        Juwrap  = FunctionWrappersWrapper(Ju, Juargtypes, (Nothing,Nothing,Nothing,Nothing,))
        Jλwrap  = FunctionWrappersWrapper(Jλ, Jλargtypes, (Nothing,Nothing,Nothing,Nothing,Nothing,Nothing,))

        new{has_full_J,typeof(fwrap),typeof(Juwrap),typeof(Jλwrap),Nothing}(
            fwrap, Juwrap, Jλwrap, nothing,
        )
    end
    function ContinuationFunction(
        f::Fi, Ju::Jui, Jλ::Jλi,
    ) where {Fi <: Function, Jui <: Function, Jλi <: Function}
        ContinuationFunction{Val{false}}(f, Ju, Jλ)
    end

    # User provided function and Jacobian wrt u and λ together
    function ContinuationFunction{has_full_J}(
        f::Fi, J::Ji,
    ) where {has_full_J <: Val{true}, Fi <: Function, Ji <: Function}
        # Construct input argument types
        λT = Float64
        VT = (
            Vector{Float64},
            SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true},
            SubArray{Float64,2,Matrix{Float64},Tuple{Base.Slice{Base.OneTo{Int}},Int},true},
        )
        MT = Matrix{Float64}
        fargtypes = (
            Tuple{VT[1],VT[1],λT},
            Tuple{VT[1],VT[2],λT},
        )
        Jargtypes = (
            Tuple{MT,VT[1],VT[1],λT},
            Tuple{MT,VT[1],VT[2],λT},
        )

        # Construct function wrappers
        fwrap   = FunctionWrappersWrapper(f, fargtypes, (Nothing,Nothing,))
        Jwrap   = FunctionWrappersWrapper(J, Jargtypes, (Nothing,Nothing,))

        new{has_full_J,typeof(fwrap),Nothing,Nothing,typeof(Jwrap)}(
            fwrap, nothing, nothing, Jwrap,
        )
    end
    function ContinuationFunction(
        f::Fi, J::Ji,
    ) where {Fi <: Function, Ji <: Function}
        ContinuationFunction{Val{true}}(f, J)
    end

    # User provided function and Jacobian wrt u and full Jacobian
    function ContinuationFunction{has_full_J}(
        f::Fi, Ju::Jui, J::Ji,
    ) where {has_full_J <: Val{true}, Fi <: Function, Jui <: Function, Ji <: Function}
        # Construct input argument types
        λT = Float64
        VT = (
            Vector{Float64},
            SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true},
        )
        MT = Matrix{Float64}
        fargtypes = (
            Tuple{VT[1],VT[1],λT},
            Tuple{VT[1],VT[2],λT},
        )
        Juargtypes = (
            Tuple{MT,VT[1],VT[1],λT},
            Tuple{MT,VT[1],VT[2],λT},
        )
        Jargtypes = (
            Tuple{MT,VT[1],VT[1],λT},
            Tuple{MT,VT[1],VT[2],λT},
        )

        # Construct function wrappers
        fwrap   = FunctionWrappersWrapper(f, fargtypes, (Nothing,Nothing,))
        Juwrap  = FunctionWrappersWrapper(Ju, Juargtypes, (Nothing,Nothing,))
        Jwrap   = FunctionWrappersWrapper(J, Jargtypes, (Nothing,Nothing,))

        new{has_full_J,typeof(fwrap),typeof(Juwrap),Nothing,typeof(Jwrap)}(
            fwrap, Juwrap, nothing, Jwrap,
        )
    end
end

struct SparseContinuationFunction{FType,JuType,JType} <: AbstractContinuationFunction
    f::FType
    Ju::JuType
    J::JType

    Ju_prototype::SparseMatrixCSC{Float64,Int}
    J_prototype::SparseMatrixCSC{Float64,Int}

    function SparseContinuationFunction(
        f::Fi, 
        Ju::Jui, Ju_prototype::SparseMatrixCSC{Float64,Int}, 
        J::Ji, J_prototype::SparseMatrixCSC{Float64,Int},
    ) where {Fi <: Function, Jui <: Function, Ji <: Function}
        # Construct input argument types
        λT = Float64
        VT = (
            Vector{Float64},
            SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true},
        )
        MT = SparseMatrixCSC{Float64,Int} 
        fargtypes = (
            Tuple{VT[1],VT[1],λT},
            Tuple{VT[1],VT[2],λT},
        )
        Juargtypes = (
            Tuple{MT,VT[1],VT[1],λT},
            Tuple{MT,VT[1],VT[2],λT},
        )
        Jargtypes = (
            Tuple{MT,VT[1],VT[1],λT},
            Tuple{MT,VT[1],VT[2],λT},
        )

        # Construct function wrappers
        fwrap   = FunctionWrappersWrapper(f, fargtypes, (Nothing,Nothing,))
        Juwrap  = FunctionWrappersWrapper(Ju, Juargtypes, (Nothing,Nothing,))
        Jwrap   = FunctionWrappersWrapper(J, Jargtypes, (Nothing,Nothing,))

        new{typeof(fwrap),typeof(Juwrap),typeof(Jwrap)}(
            fwrap, Juwrap, Jwrap, Ju_prototype, J_prototype,
        )
    end
    function SparseContinuationFunction(
        f::Fi, Ju_prototype::SparseMatrixCSC{Float64,Int}, J_prototype::SparseMatrixCSC{Float64,Int},
    ) where {Fi <: Function}
        new{typeof(f),Nothing,Nothing}(
            f, nothing, nothing, Ju_prototype, J_prototype,
        )
    end
end

# Function evaluation methods
function eval_f!(
    du, u, λ,
    cf::AbstractContinuationFunction,
)
    cf.f(du,u,λ)
    return nothing
end
function eval_f!(
    du, uλ,
    cf::AbstractContinuationFunction,
)
    # Get u and λ
    n = length(uλ)
    u = view(uλ,1:n-1)
    λ = uλ[end]

    # Eval
    eval_f!(du,u,λ,cf)
    return nothing
end

# Jacobian wrt u methods
function eval_Ju!(
    J, du, u, λ,
    cf::ContinuationFunction{hfj,F,Ju},
) where {hfj,F,Ju <: Nothing}
    # Allocate memory for full J (should only be necessary when solving the initial problem, 
    # so not too woried about unnecessary allocations at this point)
    n, m = size(J)
    Jf = zeros(n, m + 1)

    # Evaluate full Jacobian
    cf.J(Jf, du, u, λ)

    # Extract Ju
    J .= view(Jf, :, 1:m)
end
function eval_Ju!(
    J, du, u, λ,
    cf::ContinuationFunction,
)
    cf.Ju(J,du,u,λ)
    return nothing
end
function eval_Ju!(
    J, du, uλ,
    cf::ContinuationFunction,
)
    # Get u and λ
    n = length(uλ)
    u = view(uλ,1:n-1)
    λ = uλ[end]

    # Eval
    eval_Ju!(J,du,u,λ,cf)
    return nothing
end
function eval_Ju!(
    J, du, u, λ,
    cf::SparseContinuationFunction,
)
    cf.Ju(J,du,u,λ)
    return nothing
end
function eval_Ju!(
    J, du, uλ,
    cf::SparseContinuationFunction,
)
    # Get u and λ
    n = length(uλ)
    u = view(uλ,1:n-1)
    λ = uλ[end]

    # Eval
    eval_Ju!(J,du,u,λ,cf)
    return nothing
end

# Jacobian wrt u and λ methods
function eval_J!(
    J, du, u, λ,
    cf::ContinuationFunction{hfj},
) where {hfj <: Val{false}}
    # Construct views of J
    n = length(u)
    Ju = view(J, :, 1:n)
    Jλ = view(J, :, n+1)

    # Evaluate Ju and Jλ
    cf.Ju(Ju,du,u,λ)
    cf.Jλ(Jλ,du,u,λ)
    return nothing
end
function eval_J!(
    J, du, u, λ,
    cf::ContinuationFunction,
)
    cf.J(J,du,u,λ)
    return nothing
end
function eval_J!(
    J, du, uλ,
    cf::ContinuationFunction,
)
    # Get u and λ
    n = length(uλ)
    u = view(uλ,1:n-1)
    λ = uλ[end]

    eval_J!(J,du,u,λ,cf)
    return nothing
end
function eval_J!(
    J, du, u, λ,
    cf::SparseContinuationFunction,
)
    cf.J(J,du,u,λ)
    return nothing
end
function eval_J!(
    J, du, uλ,
    cf::SparseContinuationFunction,
)
    # Get u and λ
    n = length(uλ)
    u = view(uλ,1:n-1)
    λ = uλ[end]

    eval_J!(J,du,u,λ,cf)
    return nothing
end