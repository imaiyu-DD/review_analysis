using Serialization

include("Corpora.jl")
using Main.Corpora

println("reading input data...")
X_ = []
open("data/reviews.txt") do f
    for doc in eachline(f)
        X = Corpora.parse_doc(doc)
        if length(X) > 0
            push!(X_, X)
        end
    end
end

println("preprocessing train data...")
Corpora.make_corpus(X_, no_below=1, no_above=0.1)
X_ = Corpora.doc2bow(X_, T=2)

println("serializing train data...")
model_data = Dict("corpus" => Corpora.corpus, "X_" => X_)
open(io -> serialize(io, model_data), "data/train.dat", "w")

# println("deserializing train data...")
# model_data = open(deserialize, "data/train.dat")
# X_ = model_data["X_"]
# Corpora.read_corpus(model_data["corpus"])

println("executing model inferences...")
include("MGLDA.jl")
using Main.MGLDA
Φ_ = MGLDA.fit(X_, n_components=(10,20), max_iter=100, burnin=50, thinning=5, T=2)

println("serializing model parameters...")
model_params = Dict("corpus" => Corpora.corpus, "Φ_" => Φ_)
open(io -> serialize(io, model_params), "data/model.dat", "w")
