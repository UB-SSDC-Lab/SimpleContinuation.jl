
# Abstract type for pseudo-arclength
# continuation normalizations
abstract type AbstractInnerProduct end

# Unscaled dot product norm
struct StandardDotProduct <: AbstractInnerProduct end

# Scaled dot product norm
struct ScaledInnerProduct <: AbstractInnerProduct
    θ::Float64
    function ScaledInnerProduct(θ::Float64=0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        return new(θ)
    end
end

# Double Scaled dot product norm
struct DoubleScaledInnerProduct <: AbstractInnerProduct
    θ::Float64
    function DoubleScaledInnerProduct(θ::Float64=0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        return new(θ)
    end
end

# Scaled BifurcationKit norm
struct BifurcationKitInnerProduct <: AbstractInnerProduct
    θ::Float64
    function BifurcationKitInnerProduct(θ::Float64=0.5)
        if θ < 0.0 || θ > 1.0
            error("θ must be in [0,1]")
        end
        return new(θ)
    end
end

# Dot product functions
function (d::StandardDotProduct)(u1::AbstractArray, u2::AbstractArray, λ1, λ2)
    return dot(u1, u2) + λ1 * λ2
end
function (d::ScaledInnerProduct)(u1::AbstractArray, u2::AbstractArray, λ1, λ2)
    θ = d.θ
    return θ * dot(u1, u2) + (1.0 - θ) * λ1 * λ2
end
function (d::DoubleScaledInnerProduct)(u1::AbstractArray, u2::AbstractArray, λ1, λ2)
    tθ = 2.0 * d.θ
    return tθ * dot(u1, u2) + (2.0 - tθ) * λ1 * λ2
end
function (d::BifurcationKitInnerProduct)(u1::AbstractArray, u2::AbstractArray, λ1, λ2)
    n = length(u1)
    θ = d.θ
    return (θ / n) * dot(u1, u2) + (1.0 - θ) * λ1 * λ2
end
(d::AbstractInnerProduct)(u::AbstractArray, λ) = d(u, u, λ, λ)

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

function ddotdu1!(deriv, u2, d::ScaledInnerProduct)
    deriv .= d.θ .* u2
    return nothing
end
function ddotdu2!(deriv, u1, d::ScaledInnerProduct)
    deriv .= d.θ .* u1
    return nothing
end
function ddotdλ1(λ2, d::ScaledInnerProduct)
    return (1.0 - d.θ) * λ2
end
function ddotdλ2(λ1, d::ScaledInnerProduct)
    return (1.0 - d.θ) * λ1
end

function ddotdu1!(deriv, u2, d::DoubleScaledInnerProduct)
    deriv .= 2.0 * d.θ .* u2
    return nothing
end
function ddotdu2!(deriv, u1, d::DoubleScaledInnerProduct)
    deriv .= 2.0 * d.θ .* u1
    return nothing
end
function ddotdλ1(λ2, d::DoubleScaledInnerProduct)
    return 2.0 * (1.0 - d.θ) * λ2
end
function ddotdλ2(λ1, d::DoubleScaledInnerProduct)
    return 2.0 * (1.0 - d.θ) * λ1
end

function ddotdu1!(deriv, u2, d::BifurcationKitInnerProduct)
    st = d.θ / length(u2)
    deriv .= st .* u2
    return nothing
end
function ddotdu2!(deriv, u1, d::BifurcationKitInnerProduct)
    st = d.θ / length(u1)
    deriv .= st .* u1
    return nothing
end
function ddotdλ1(λ2, d::BifurcationKitInnerProduct)
    return (1.0 - d.θ) * λ2
end
function ddotdλ2(λ1, d::BifurcationKitInnerProduct)
    return (1.0 - d.θ) * λ1
end

# Normalization constraints
function palc_norm(δu, δu0, δλ, δλ0, ds, d::AbstractInnerProduct)
    return d(δu, δu0, δλ, δλ0) - ds
end

# The norm partial wrt δu
function palc_norm_dδu!(dnorm_du, δu0, d::AbstractInnerProduct)
    return ddotdu1!(dnorm_du, δu0, d)
end

# The norm partial wrt λ
function palc_norm_dδλ(δλ0, d::AbstractInnerProduct)
    return ddotdλ1(δλ0, d)
end
