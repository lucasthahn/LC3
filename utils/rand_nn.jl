using Flux

struct RandomNN{T<:AbstractFloat}
	model::Flux.Chain

	function RandomNN{T}(input, numfeatures) where T<:AbstractFloat
		m = Chain(Dense(input, 164, tanh), Dense(164, numfeatures))
		new{T}(m)
	end
end

function (nn::RandomNN)(x::AbstractArray)
	z = nn.model(x)
   z
end

(nn::RandomNN{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = nn.model(T.(x))

function (nn::RandomNN{T})(z::AbstractArray{T, N},
                           x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
   z = nn.model(T.(x))
   z
end
