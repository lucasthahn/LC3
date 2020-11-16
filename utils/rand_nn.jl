using Flux

hidden_1 = 64
hidden_2 = 128

struct RandomNN{T<:AbstractFloat}
	model::Flux.Chain

	function RandomNN{T}(input, numfeatures) where T<:AbstractFloat
		m = Chain(Dense(input, hidden_1, tanh),
					 Dense(hidden_1, hidden_2, tanh),
					 Dense(hidden_2, numfeatures))
		new{T}(m)
	end
end

function (nn::RandomNN)(x::AbstractArray)
	nn.model(x)
end

(nn::RandomNN{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = nn.model(T.(x))

function (nn::RandomNN{T})(z::AbstractArray{T, N},
                           x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
   z = nn.model(T.(x))
   z
end
