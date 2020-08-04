FROM julia:1.0.5-buster

LABEL maintainer="Yusaku Imai <imai.yu@dentsudigital.co.jp>"

RUN apt-get -y update && \
    apt-get install -y sudo mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file patch

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    ./bin/install-mecab-ipadic-neologd -n -y --prefix /usr/local/lib/mecab/dic/mecab-ipadic-neologd

RUN git clone https://github.com/imaiyu-DD/review_analysis.git
RUN julia -e 'using Pkg; Pkg.add("StatsBase"); Pkg.add("MeCab")'
