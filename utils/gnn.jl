using Flux
using Random
using LightGraphs
using GeometricFlux

hidden = 64
nodes = 100

struct GNN{T<:AbstractFloat}
	model::Flux.Chain
	conv_buf::Matrix{T}
	buffer::Matrix{T}

	function GNN{T}(input, numfeatures) where T<:AbstractFloat
		g = SimpleDiGraph(nodes, rand(1:nodes))
		c = randn(T, 1, nodes)
		b = randn(T, input, nodes)
		m = Chain(GCNConv(g, input=>hidden, tanh), GCNConv(g, hidden=>numfeatures))
		new{T}(m, c, b)
	end
end

function (gnn::GNN)(x::AbstractArray)
	mul!(gnn.buffer, x, gnn.conv_buf)
	vec(gnn.model(gnn.buffer))
end

function (gnn::GNN{T})(z::AbstractArray{T, N},
                       x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
	mul!(gnn.buffer, x, gnn.conv_buf)
	z = vec(gnn.model(gnn.buffer))
	z
end
