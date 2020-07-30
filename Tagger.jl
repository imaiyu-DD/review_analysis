module Tagger

using MeCab
using LinearAlgebra

struct model_params
    parser::MeCab.Mecab
    Φ_::Array{Float64,2}
    corpus::Vector{String}
    tags::Dict{Int64,String}
end

function set_params(Φ_::Array, corpus::Vector, tags::Dict)
    global params = model_params(Mecab("-d/usr/local/lib/mecab/dic/mecab-ipadic-neologd"), Φ_, corpus, tags)
end

function tagging(doc::String; ɛ::Float64=0.05, debug=false)
    p_ = score(doc2bow(doc))

    if debug
        return [(params.tags[topicid], p_[topicid]) for topicid in keys(params.tags) if p_[topicid] >= ɛ]
    else
        return [params.tags[topicid] for topicid in keys(params.tags) if p_[topicid] >= ɛ]
    end
end

function doc2bow(doc::String)
    @assert isdefined(params, :parser)

    bow = zeros(Int, length(params.corpus))
    for result in parse(params.parser, doc)
        feats = split(result.feature, ",")
        if length(feats) > 7 && feats[7] in params.corpus
            ind = findfirst(isequal(feats[7]), params.corpus)
            bow[ind] += 1
        end
    end

    return bow
end

function cossim(x_::Vector, y_::Vector)
    return dot(x_, y_) / (norm(x_, 2) * norm(y_, 2))
end

function score(bow::Vector)
    @assert isdefined(params, :Φ_)

    return [cossim(bow, params.Φ_[k,:]) for k in 1:size(params.Φ_, 1)]
end

function score(doc::String, topicid::Int)
    @assert isdefined(params, :Φ_)

    return cossim(doc2bow(doc), params.Φ_[topicid,:])
end

end # module
