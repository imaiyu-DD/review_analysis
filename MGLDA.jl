module MGLDA

using StatsBase
using Random
using ..Corpora

const α_loc = 0.5
const α_gl  = 0.5
const β     = 0.1
const γ     = 0.5
const a     = 1.0
const b     = 1.0

function fit(X_::Array; n_components::Tuple=(10,20), max_iter::Int=150, burnin::Int=100, thinning::Int=10, T::Int=2)
    @assert isdefined(Corpora, :corpus)
    @assert n_components[1] > 0 && n_components[2] > 0
    @assert max_iter >= burnin > 0 && thinning > 0 && T > 0

    # multi-grain LDA
    n_words = length(Corpora.corpus)
    cgs(X_, n_components, max_iter, burnin, thinning, T, n_words)
end

# estimate model parameters with collapsed Gibbs sampling
function cgs(X_::Array, n_components::Tuple, max_iter::Int, burnin::Int, thinning::Int, T::Int, n_words::Int)
    n_docs = length(X_)

    r_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]
    z_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]
    v_dsn = [[zeros(Int, N_ds) for (N_ds, X) in X_[d][1]] for d in 1:n_docs]

    N_dsv = [zeros(Int, length(X_[d][1]), length(X_[d][2])) for d in 1:n_docs]

    N_dvr = [zeros(Int, length(X_[d][2]), 2) for d in 1:n_docs]
    N_dv  = [zeros(Int, length(X_[d][2])) for d in 1:n_docs]

    N_loc_kw  = zeros(Int, n_components[1], n_words)
    N_loc_k   = zeros(Int, n_components[1])
    N_loc_dvk = [zeros(Int, length(X_[d][2]), n_components[1]) for d in 1:n_docs]
    N_loc_dv  = [zeros(Int, length(X_[d][2])) for d in 1:n_docs]

    N_gl_kw = zeros(Int, n_components[2], n_words)
    N_gl_k  = zeros(Int, n_components[2])
    N_gl_dk = zeros(Int, n_docs, n_components[2])
    N_gl_d  = zeros(Int, n_docs)

    Random.seed!(0)
    samples_N_loc_kw = []
    samples_N_gl_kw  = []
    @inbounds for iter in 1:max_iter
        @inbounds for d in 1:n_docs
            @inbounds for (s, (N_ds, X)) in enumerate(X_[d][1])
                n = 0
                @inbounds for (w, N_dsw) in X
                    @inbounds for _ in 1:N_dsw
                        n += 1

                        # remove w_dsn’s statistics
                        if z_dsn[d][s][n] != 0
                            r = r_dsn[d][s][n]
                            z = z_dsn[d][s][n]
                            v = v_dsn[d][s][n]
                            N_dsv[d][s,v] -= 1
                            N_dvr[d][v,r] -= 1
                            N_dv[d][v]    -= 1

                            # loc
                            if r == 1
                                N_loc_kw[z,w]     -= 1
                                N_loc_k[z]        -= 1
                                N_loc_dvk[d][v,z] -= 1
                                N_loc_dv[d][v]    -= 1

                            # gl
                            elseif r == 2
                                N_gl_kw[z,w] -= 1
                                N_gl_k[z]    -= 1
                                N_gl_dk[d,z] -= 1
                                N_gl_d[d]    -= 1
                            end
                        end

                        log_p_loc_vz = zeros(T, n_components[1])
                        log_p_gl_vz  = zeros(T, n_components[2])
                        @inbounds for t in 1:T
                            v = s + t - 1

                            # loc
                            log_p_loc_vz[t,:] .+= log(N_dsv[d][s,v] + γ) # - log(N_ds + γ * T)
                            log_p_loc_vz[t,:] .+= log(N_dvr[d][v,1] + a) # - log(X_[d][2][v] + a + b)
                            for k in 1:n_components[1]
                                log_p_loc_vz[t,k] += log(N_loc_kw[k,w] + β) - log(N_loc_k[k] + β * n_words)
                                log_p_loc_vz[t,k] += log(N_loc_dvk[d][v,k] + α_loc) - log(N_loc_dv[d][v] + α_loc * n_components[1])
                            end

                            # gl
                            log_p_gl_vz[t,:] .+= log(N_dsv[d][s,v] + γ) # - log(N_ds + γ * T)
                            log_p_gl_vz[t,:] .+= log(N_dvr[d][v,2] + b) # - log(X_[d][2][v] + a + b)
                            for k in 1:n_components[2]
                                log_p_gl_vz[t,k] += log(N_gl_kw[k,w] + β) - log(N_gl_k[k] + β * n_words)
                                log_p_gl_vz[t,k] += log(N_gl_dk[d,k] + α_gl) - log(N_gl_d[d] + α_gl * n_components[2])
                            end
                        end

                        # sample v_dsn, r_dsn, and z_dsn
                        log_p_vz = collect(Iterators.flatten(hcat(log_p_loc_vz, log_p_gl_vz)))
                        p_vz     = exp.(log_p_vz .- maximum(log_p_vz))

                        ind_dsn        = sample(1:length(p_vz), Weights(p_vz / sum(p_vz)))
                        v_dsn[d][s][n] = s + rem(ind_dsn - 1, T)

                        if ind_dsn <= T * n_components[1]
                            r_dsn[d][s][n] = 1
                            z_dsn[d][s][n] = div(ind_dsn - 1, T) + 1
                        else
                            r_dsn[d][s][n] = 2
                            z_dsn[d][s][n] = div(ind_dsn - T * n_components[1] - 1, T) + 1
                        end

                        # add w_dsn’s statistics
                        r = r_dsn[d][s][n]
                        z = z_dsn[d][s][n]
                        v = v_dsn[d][s][n]
                        N_dsv[d][s,v] += 1
                        N_dvr[d][v,r] += 1
                        N_dv[d][v]    += 1

                        # loc
                        if r == 1
                            N_loc_kw[z,w]     += 1
                            N_loc_k[z]        += 1
                            N_loc_dvk[d][v,z] += 1
                            N_loc_dv[d][v]    += 1

                        # gl
                        elseif r == 2
                            N_gl_kw[z,w] += 1
                            N_gl_k[z]    += 1
                            N_gl_dk[d,z] += 1
                            N_gl_d[d]    += 1
                        end
                    end
                end
            end
        end

        if iter >= burnin && rem(iter - burnin, thinning) == 0
            push!(samples_N_loc_kw, N_loc_kw)
            push!(samples_N_gl_kw, N_gl_kw)
        end
    end

    return posteriori_estimation(mean(samples_N_loc_kw), mean(samples_N_gl_kw), n_words)
end

function posteriori_estimation(N_loc_kw::Array, N_gl_kw::Array, n_words::Int)
    Φ_loc = (N_loc_kw .+ β) ./ (sum(N_loc_kw, dims=1) .+ β * n_words)
    Φ_gl  = (N_gl_kw .+ β) ./ (sum(N_gl_kw, dims=1) .+ β * n_words)
    Φ_    = vcat(Φ_loc, Φ_gl)

    return Φ_
end

end # module
