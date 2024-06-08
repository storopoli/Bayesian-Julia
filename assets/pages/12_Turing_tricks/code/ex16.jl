# This file was generated, do not modify it. # hide
τ = summarystats(chain_ncp)[:τ, :mean]
αⱼ = mapslices(
    x -> x * τ, chain_ncp[:, namesingroup(chain_ncp, :zⱼ), :].value.data; dims=[2]
)
chain_ncp_reconstructed = hcat(
    MCMCChains.resetrange(chain_ncp), Chains(αⱼ, ["αⱼ[$i]" for i in 1:length(unique(idx))])
)