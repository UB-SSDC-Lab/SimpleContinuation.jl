
# Abstract type for pseudo-arclength 
# continuation normalizations
abstract type AbstractNorm end

# Unscaled dot product norm
struct StandardNorm <: AbstractNorm end

# Scaled dot product norm
struct ScaledNorm <: AbstractNorm 
    θ::Float64
    function ScaledNorm(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# Double Scaled dot product norm
struct DoubleScaledNorm <: AbstractNorm 
    θ::Float64
    function DoubleScaledNorm(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# Scaled BifurcationKit norm
struct BifurcationKitNorm <: AbstractNorm 
    θ::Float64
    function BifurcationKitNorm(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# The norm as defined by norm type
function pnorm(
    δu, δu0, δλ, δλ0, 
    ds, norm::StandardNorm,
)
    return dot(δu, δu0) + δλ*δλ0 - ds
end
function pnorm(
    δu, δu0, δλ, δλ0, 
    ds, norm::ScaledNorm,
)
    θ = norm.θ
    return θ*dot(δu, δu0) + (1.0 - θ)*δλ*δλ0 - ds
end
function pnorm(
    δu, δu0, δλ, δλ0, 
    ds, norm::DoubleScaledNorm,
)
    θ = 2.0*norm.θ
    return θ*dot(δu, δu0) + (1.0 - θ)*δλ*δλ0 - ds
end
function pnorm(
    δu, δu0, δλ, δλ0, 
    ds, norm::BifurcationKitNorm,
) 
    θ = norm.θ
    t1 = θ / length(δu0)
    t2 = 1.0 - θ
    return t1*dot(δu, δu0) + t2*δλ*δλ0 - ds
end

# The norm partial wrt u
function dpnorm_du!(    
    dnorm_du,
    δu, δu0, δλ, δλ0, 
    norm::StandardNorm,
)
    for i in eachindex(dnorm_du)
        dnorm_du[i] = δu0[i]
    end
    return nothing
end
function dpnorm_du!(    
    dnorm_du,
    δu, δu0, δλ, δλ0, 
    norm::ScaledNorm,
)
    θ = norm.θ
    for i in eachindex(dnorm_du)
        dnorm_du[i] = θ*δu0[i]
    end
    return nothing
end
function dpnorm_du!(    
    dnorm_du,
    δu, δu0, δλ, δλ0, 
    norm::DoubleScaledNorm,
)
    θ = 2.0*norm.θ
    for i in eachindex(dnorm_du)
        dnorm_du[i] = θ*δu0[i]
    end
    return nothing
end
function dpnorm_du!(    
    dnorm_du,
    δu, δu0, δλ, δλ0, 
    norm::BifurcationKitNorm,
)
    θ = norm.θ
    t1 = θ / length(δu0)
    for i in eachindex(dnorm_du)
        dnorm_du[i] = t1*δu0[i]
    end
    return nothing
end

# The norm partial wrt λ
function dpnorm_dλ(    
    δu, δu0, δλ, δλ0, 
    norm::StandardNorm,
)
    return δλ0
end
function dpnorm_dλ(    
    δu, δu0, δλ, δλ0, 
    norm::ScaledNorm,
)
    θ = norm.θ
    return (1.0 - θ)*δλ0
end
function dpnorm_dλ(    
    δu, δu0, δλ, δλ0, 
    norm::DoubleScaledNorm,
)
    θ = 2.0*norm.θ
    return (1.0 - θ)*δλ0
end
function dpnorm_dλ(    
    δu, δu0, δλ, δλ0, 
    norm::BifurcationKitNorm,
)
    θ = norm.θ
    return (1.0 - θ)*δλ0
end