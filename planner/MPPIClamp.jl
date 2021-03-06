"""
This file is for computing MPPI planning with actions appropriately clamped.
Base code borrowed from LyceumAI.jl https://github.com/Lyceum/LyceumAI.jl
    envs: these are for parallel computing of shootings.  In our work, these envs represent learned models.
"""

using StaticArrays
using Base: @propagate_inbounds, require_one_based_indexing

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix

struct MPPIClamp{DT<:AbstractFloat,nu,Covar<:AbsMat{DT},Value,Env,Init,Obs,State}
    # MPPIClamp parameters
    K::Int
    H::Int
    lambda::DT
    gamma::DT
    value::Value
    envs::Vector{Env} # one per thread
    initfn!::Init

    # internal
    noise::Array{DT,3}
    covar_ul::Covar
    meantrajectory::Matrix{DT}
    clamps::SMatrix{2, nu, DT}
    trajectorycosts::Vector{DT}
    obsbuffers::Vector{Obs}
    statebuffers::Vector{State}

    function MPPIClamp{DT}(
        env_tconstructor,
        K::Integer,
        H::Integer,
        covar::AbstractMatrix{<:Real},
        lambda::Real,
        gamma::Real,
        value,
        initfn!,
        clamps
    ) where {DT<:AbstractFloat}
        envs = [e for e in env_tconstructor(Threads.nthreads())]

        ssp = statespace(first(envs))
        asp = actionspace(first(envs))
        osp = obsspace(first(envs))
        nu = length(asp)

        if !(asp isa Shapes.AbstractVectorShape)
            throw(ArgumentError("actionspace(env) must be a Shape.AbstractVectorShape"))
        end
        K > 0 || error("K must be > 0. Got $K.")
        H > 0 || error("H must be > 0. Got $H.")
        0 < lambda <= 1 || throw(ArgumentError("lambda must be in interval (0, 1]"))
        0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))

        covar_ul = convert(AbsMat{DT}, cholesky(covar).UL)
        meantrajectory = zeros(DT, asp, H)
        if clamps == nothing
            clamps = SMatrix{2, nu, DT}(vcat(fill(typemin(DT), 1, nu), fill(typemax(DT), 1, nu)))
        elseif typeof(clamps) <: SArray == false
            clamps = SMatrix{2, nu, DT}(clamps)
        end
        trajectorycosts = zeros(DT, K)
        noise = zeros(DT, asp, H, K)
        obsbuffers = [allocate(osp) for _ = 1:Threads.nthreads()]
        statebuffers = [allocate(ssp) for _ = 1:Threads.nthreads()]

        new{
            DT,
            nu,
            typeof(covar_ul),
            typeof(value),
            eltype(envs),
            typeof(initfn!),
            eltype(obsbuffers),
            eltype(statebuffers),
        }(
            K,
            H,
            lambda,
            gamma,
            value,
            envs,
            initfn!,
            noise,
            covar_ul,
            meantrajectory,
            clamps,
            trajectorycosts,
            obsbuffers,
            statebuffers
        )
    end
end

using DocStringExtensions
"""
    $(TYPEDEF)
    MPPIClamp{DT<:AbstractFloat}(args...; kwargs...) -> MPPIClamp
    MPPIClamp(args...; kwargs...) -> MPPIClamp
Construct an instance of  `MPPIClamp` with `args` and `kwargs`, where `DT <: AbstractFloat` is
the element type used for pre-allocated buffers, which defaults to Float32.
In the following explanation of the `MPPIClamp` constructor, we use the
following notation:
- `U::Matrix`: the canonical control vector ``(u_{1}, u_{2}, \\dots, u_{H})``, where
    `size(U) == (length(actionspace(env)), H)`.
# Keywords
- `env_tconstructor`: a function with signature `env_tconstructor(n)` that returns `n`
    instances of `T`, where `T <: AbstractEnvironment`.
- `H::Integer`: Length of sampled trajectories.
- `K::Integer`: Number of trajectories to sample.
- `covar::AbstractMatrix`: The covariance matrix for the Normal distribution from which
    control pertubations are sampled from.
- `gamma::Real`: Reward discount, applied as `gamma^(t - 1) * reward[t]`.
- `lambda::Real`: Temperature parameter for the exponential reweighting of sampled
    trajectories. In the limit that lambda approaches 0, `U` is set to the highest reward
    trajectory. Conversely, as `lambda` approaches infinity, `U` is computed as the
    unweighted-average of the samples trajectories.
- `value`: a function mapping observations to scalar rewards, with the signature
    `value(obs::AbstractVector) --> reward::Real`
- `initfn!`: A function with the signature `initfn!(U::Matrix)` used for
    re-initializing `U` after shifting it. Defaults to setting the last
    element of `U` to 0.
"""
function MPPIClamp{DT}(;
    env_tconstructor,
    covar,
    lambda,
    K,
    H,
    gamma = 1,
    value = zerofn,
    initfn! = default_initfn!,
    clamps = nothing 
) where {DT<:AbstractFloat}
    MPPIClamp{DT}(env_tconstructor, K, H, covar, lambda, gamma, value, initfn!, clamps)
end

MPPIClamp(args...; kwargs...) = MPPIClamp{Float32}(args...; kwargs...)

function _clampctrl!(actions::Array{DT, 3}, clamps::AbstractArray) where DT<:AbstractFloat
    nu, H, K = size(actions)
    for k = 1:K, t = 1:H, u = 1:nu
        @inbounds actions[u, t, k] = clamp(actions[u,t,k], clamps[1,u], clamps[2,u]) 
    end
end

"""
    $(TYPEDSIGNATURES)
Resets the canonical control vector to zeros.
"""
LyceumBase.reset!(m::MPPIClamp) = (fill!(m.meantrajectory, 0); m)

"""
    $(SIGNATURES)
Starting from the environment's `state`, perform one step of the MPPIClamp algorithm and
store the resulting action in `action`. The trajectory sampling portion of MPPIClamp is
done in parallel using `nthreads` threads.
"""
@propagate_inbounds function LyceumBase.getaction!(
    action::AbstractVector,
    state,
    m::MPPIClamp{DT,nu};
    nthreads::Integer = Threads.nthreads(),
) where {DT,nu}
    @boundscheck begin
        length(action) == nu || throw(ArgumentError("Expected action vector of length $nu. Got: $(length(action))"))
        require_one_based_indexing(action)
    end

    nthreads = min(m.K, nthreads)
    step!(m, state, nthreads)
    @inbounds copyto!(action, uview(m.meantrajectory, :, 1))
    shiftcontrols!(m)
    return action
end

function LyceumBase.step!(m::MPPIClamp{DT,nu}, s, nthreads) where {DT,nu}
    randn!(m.noise)
    lmul!(m.covar_ul, reshape(m.noise, (nu, :)))
    m.noise .+= m.meantrajectory # noise is now the mean + noise = actions
    _clampctrl!(m.noise, m.clamps)
    if nthreads == 1
        # short circuit
        threadstep!(m, s, 1:m.K)
    else
        kranges = splitrange(m.K, nthreads)
        @sync for i = 1:nthreads
            Threads.@spawn threadstep!(m, s, kranges[i])
        end
    end
    combinetrajectories!(m)
end

function threadstep!(m::MPPIClamp, s, krange)
    tid = Threads.threadid()
    for k in krange
        perturbedrollout!(m, s, k, tid)
    end
end

function perturbedrollout!(m::MPPIClamp{DT,nu}, state, k, tid) where {DT,nu}
    env = m.envs[tid]
    obsbuf = m.obsbuffers[tid]
    statebuf = m.statebuffers[tid]
    mean = m.meantrajectory
    noise = m.noise

    setstate!(env, state)
    getobs!(obsbuf, env)
    discountedreward = zero(DT)
    discountfactor = one(DT)
    @uviews noise @inbounds for t = 1:m.H
        action_t = SVector{nu,DT}(view(noise, :, t, k))
        setaction!(env, action_t)
        step!(env)
        getobs!(obsbuf, env)
        getstate!(statebuf, env)
        reward = getreward(statebuf, action_t, obsbuf, env)
        discountedreward += reward * discountfactor
        discountfactor *= m.gamma
    end 
    getobs!(obsbuf, env)
    terminalvalue = convert(DT, m.value(obsbuf))
    @inbounds m.trajectorycosts[k] = -(discountedreward + terminalvalue * discountfactor)
    return m
end

function combinetrajectories!(m::MPPIClamp{DT,nu}) where {DT,nu}
    costs = m.trajectorycosts
    beta = minimum(costs)
    eta = zero(DT)
    for k = 1:m.K
        @inbounds costs[k] = softcost = exp((beta - costs[k]) / m.lambda)
        eta += softcost
    end
    costs ./= eta
    m.meantrajectory .= zero(DT)
    for k = 1:m.K, t = 1:m.H, u = 1:nu
        @inbounds m.meantrajectory[u, t] += costs[k] * m.noise[u, t, k]
    end
    m
end

function shiftcontrols!(m::MPPIClamp{DT,nu}) where {DT,nu}
    for t = 2:m.H, u = 1:nu
        @inbounds m.meantrajectory[u, t-1] = m.meantrajectory[u, t]
    end
    m.initfn!(m.meantrajectory)
    m
end

@inline function default_initfn!(meantraj)
    @uviews meantraj @inbounds begin
        lastcontrols = view(meantraj, :, size(meantraj, 2))
        fill!(lastcontrols, 0)
    end
end
