module Corpora

using StatsBase
using MeCab
mecab = Mecab("-d/usr/local/lib/mecab/dic/mecab-ipadic-neologd")

function parse_doc(doc::String)
    ret = []
    doc = replace(doc, "\\n" => "。")
    doc = replace(doc, "．" => "。")
    doc = replace(doc, "!" => "。")
    doc = replace(doc, "?" => "。")
    doc = replace(doc, "！" => "。")
    doc = replace(doc, "？" => "。")
    for sentence in split(doc, "。")
        s_ = []
        for result in parse(mecab, string(sentence, "。"))
            if startswith(result.feature, "形容詞,自立") || startswith(result.feature, "名詞,サ変接続") || startswith(result.feature, "名詞,ナイ形容詞語幹") || startswith(result.feature, "名詞,形容動詞語幹") || startswith(result.feature, "名詞,一般") || startswith(result.feature, "名詞,固有名詞")
                feats = split(result.feature, ",")
                if length(feats) > 7
                    if !occursin(r"[0-9]", feats[7])
                        push!(s_, feats[7])
                    elseif occursin(r"[0-9]+円", feats[7])
                        push!(s_, "X円")
                    elseif occursin(r"[0-9]+分", feats[7])
                        push!(s_, "X分")
                    elseif occursin(r"[0-9]+階", feats[7])
                        push!(s_, "X階")
                    end
                end
            end
        end

        if length(s_) > 0
            push!(ret, s_)
        end
    end

    return ret
end

function read_corpus(V_::Array)
    global corpus = V_
end

function make_corpus(X_::Array; no_below::Int=0, no_above::Float64=1.0)
    dict1 = filter((k, w) -> w > no_below, countmap(flatten(X_)))
    dict2 = filter((k, w) -> w / length(X_) > no_above, countmap(flatten(X_, uniq=true)))
    global corpus = collect(setdiff(keys(dict1), keys(dict2)))
end

function flatten(X_::Array; uniq::Bool=false)
    if uniq
        return collect(Iterators.flatten([unique(Iterators.flatten(X)) for X in X_]))
    else
        return collect(Iterators.flatten(Iterators.flatten(X_)))
    end
end

function doc2bow(X_::Array; T::Int=2)
    @assert isdefined(Corpora, :corpus)

    ret = Vector{Any}()
    for X in X_
        s_ = Vector{Any}()
        for s in X
            bow  = filter((k, w) -> k ∈ corpus, countmap(s))
            N_ds = sum([w for (k, w) in bow])
            push!(s_, (N_ds, [(findfirst(isequal(k), corpus), w) for (k, w) in bow]))
        end

        N_dv = Vector{Int}()
        for i in 2-T:length(s_)
            push!(N_dv, sum([s_[j][1] for j in clamp(i,1,length(s_)):clamp(i+T-1,1,length(s_))]))
        end

        push!(ret, (s_, N_dv))
    end

    return ret
end

end
