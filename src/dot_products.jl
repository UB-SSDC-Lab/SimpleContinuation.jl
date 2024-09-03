
# Abstract type for pseudo-arclength 
# continuation normalizations
abstract type AbstractDotProduct end

# Unscaled dot product norm
struct StandardDotProduct <: AbstractDotProduct end

# Scaled dot product norm
struct ScaledDotProduct <: AbstractDotProduct
    θ::Float64
    function ScaledDotProduct(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# Double Scaled dot product norm
struct DoubleScaledDotProduct <: AbstractDotProduct 
    θ::Float64
    function DoubleScaledDotProduct(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# Scaled BifurcationKit norm
struct BifurcationKitDotProduct <: AbstractDotProduct
    θ::Float64
    function BifurcationKitDotProduct(θ::Float64 = 0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        new(θ)
    end
end

# Dot product functions
function (d::StandardDotProduct)(
    u1::AbstractArray, u2::AbstractArray, λ1, λ2,
)
    return dot(u1, u2) + λ1*λ2
end
function (d::ScaledDotProduct)(
    u1::AbstractArray, u2::AbstractArray, λ1, λ2,
)
    θ = d.θ
    return θ*dot(u1, u2) + (1.0 - θ)*λ1*λ2
end
function (d::DoubleScaledDotProduct)(
    u1::AbstractArray, u2::AbstractArray, λ1, λ2,
)
    tθ = 2.0*d.θ
    return tθ*dot(u1, u2) + (2.0 - tθ)*λ1*λ2
end
function (d::BifurcationKitDotProduct)(
    u1::AbstractArray, u2::AbstractArray, λ1, λ2,
)
    n = length(u1)
    θ = d.θ
    return (θ / n)*dot(u1, u2) + (1.0 - θ)*λ1*λ2
end
(d::AbstractDotProduct)(u::AbstractArray, λ) = d(u, u, λ, λ)

# Dot product partials
function ddotdu1!(deriv, u2, d::StandardDotProduct)
    deriv .= u2
    return nothing
end
function ddotdu2!(deriv, u1, d::StandardDotProduct)
    deriv .= u1
    return nothing
end
function ddotdλ1(λ2, d::StandardDotProduct)
    return λ2 
end
function ddotdλ2(λ1, d::StandardDotProduct)
    return λ1
end

function ddotdu1!(deriv, u2, d::ScaledDotProduct)
    deriv .= d.θ.*u2
    return nothing
end
function ddotdu2!(deriv, u1, d::ScaledDotProduct)
    deriv .= d.θ.*u1
    return nothing
end
function ddotdλ1(λ2, d::ScaledDotProduct)
    return (1.0 - d.θ)*λ2 
end
function ddotdλ2(λ1, d::ScaledDotProduct)
    return (1.0 - d.θ)*λ1
end

function ddotdu1!(deriv, u2, d::DoubleScaledDotProduct)
    deriv .= 2.0*d.θ.*u2
    return nothing
end
function ddotdu2!(deriv, u1, d::DoubleScaledDotProduct)
    deriv .= 2.0*d.θ.*u1
    return nothing
end
function ddotdλ1(λ2, d::DoubleScaledDotProduct)
    return 2.0*(1.0 - d.θ)*λ2 
end
function ddotdλ2(λ1, d::DoubleScaledDotProduct)
    return 2.0*(1.0 - d.θ)*λ1
end

function ddotdu1!(deriv, u2, d::BifurcationKitDotProduct)
    st = d.θ / length(u2)
    deriv .= st .* u2
    return nothing
end
function ddotdu2!(deriv, u1, d::BifurcationKitDotProduct)
    st = d.θ / length(u1)
    deriv .= st .* u1
    return nothing
end
function ddotdλ1(λ2, d::BifurcationKitDotProduct)
    return (1.0 - d.θ)*λ2 
end
function ddotdλ2(λ1, d::BifurcationKitDotProduct)
    return (1.0 - d.θ)*λ1
end

# Normalization constraints
function palc_norm(
    δu, δu0, δλ, δλ0, ds, d::AbstractDotProduct,
)
    return d(δu,δu0,δλ,δλ0) - ds
end

# The norm partial wrt δu
function palc_norm_dδu!(    
    dnorm_du, δu0, d::AbstractDotProduct,
)
    ddotdu1!(dnorm_du, δu0, d)
end

# The norm partial wrt λ
function palc_norm_dδλ(    
    δλ0, d::AbstractDotProduct,
)
    return ddotdλ1(δλ0, d)
end