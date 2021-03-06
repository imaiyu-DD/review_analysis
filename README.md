# review_analysis
## 分析環境
```sh
$ julia -v
julia version 1.0.5
```

```julia
julia> using Pkg
julia> Pkg.installed()
"StatsBase" => v"0.33.0"
"MeCab"     => v"0.2.0"
```

MeCab辞書: `/usr/local/lib/mecab/dic/mecab-ipadic-neologd`

## モデル学習
ホテルの口コミデータ data/reviews.txt から [Multi-grain Topic Model](https://www.researchgate.net/publication/1906122_Modeling_Online_Reviews_with_Multi-grain_Topic_Models) の学習を以下で行う
```sh
$ julia model_building.jl
```
学習が終わると data/model.dat に分析のためのモデルパラメータファイルが吐き出される

## トピックの可視化

モデルパラメータから各トピックを以下で可視化する

```julia
julia> using Serialization
julia> model_params = open(deserialize, "data/model.dat")
julia> for k in 1:size(model_params["Φ_"],1) println([model_params["corpus"][w] for w in sortperm(model_params["Φ_"][k, :], rev=true)[1:10]]) end
1 ["料理", "おいしい", "ビュッフェ", "種類", "和食", "ルームサービス", "バイキング", "味", "メニュー", "夕食"]
2 ["ベッド", "風呂", "シャワー", "bathtab", "トイレ", "寝心地", "大きい", "テレビ", "ベット", "狭い"]
3 ["東京タワー", "眺め", "窓", "スカイツリー", "眺望", "高層", "きれい", "綺麗", "目", "海"]
4 ["プール", "気分", "ジャグジー", "リラックス", "無料", "スパ", "ジム", "空間", "SPA!", "贅沢"]
5 ["値段", "高い", "プラン", "料金", "ツイン", "安い", "価格", "デラックス", "得", "X円"]
6 ["丁寧", "親切", "気持ち", "笑顔", "素晴らしい", "接客", "方々", "気持ちよい", "大変", "印象"]
7 ["家族", "子供", "仕事", "機会", "旅行", "是非", "一泊", "滞在", "友人", "二人"]
8 ["お台場", "食事", "場所", "コンビニ", "観光", "店", "おすすめ", "近い", "周辺", "お勧め"]
9 ["アメニティ", "充実", "用意", "嬉しい", "アメニティー", "タオル", "清潔", "セット", "シャンプー", "設備"]
10 ["フロア", "エグゼクティブ", "クラブ", "カクテル", "タイム", "バー", "X階", "コーヒー", "席", "飲み物"]
11 ["誕生日", "素敵", "ケーキ", "母", "用意", "お祝い", "記念日", "嬉しい", "写真", "プレゼント"]
12 ["気", "音", "エレベーター", "廊下", "カードキー", "うるさい", "必要", "隣", "階", "清掃"]
13 ["客室", "改装", "メイン", "タワー", "ニューオータニ", "前回", "禅", "ロビー", "ポイント", "ネット"]
14 ["電話", "確認", "案内", "説明", "連絡", "女性", "お願い", "無い", "最初", "男性"]
15 ["東京駅", "Shangri-La", "高級ホテル", "好き", "香り", "雰囲気", "ロビー", "エントランス", "内装", "入口"]
16 ["直結", "羽田空港", "ゆりかもめ", "リムジンバス", "アクセス", "空港", "無料", "羽田", "お台場", "交通"]
17 ["荷物", "チェックアウト", "到着", "スムーズ", "客", "タクシー", "笑", "ロビー", "団体", "列"]
18 ["禁煙", "喫煙", "案内", "変更", "気", "希望", "臭い", "遅い", "仕方", "安心"]
19 ["新宿", "新宿駅", "遠い", "X分", "シャトルバス", "近い", "雨", "西口", "ロビー", "徒歩"]
20 ["雰囲気", "和", "モダン", "外国人", "静か", "和風", "ビジネスホテル", "障子", "インテリア", "外国"]
21 ["ヒルトン", "日航", "会員", "アップグレード", "特典", "ヒルトン東京", "スイート", "メンバー", "エグゼクティブ", "時代"]
22 ["人", "他", "必要", "すごい", "自分", "非常", "一つ", "地方", "注意", "少ない"]
23 ["滞在", "不便", "仕事", "問題", "移動", "都内", "台場", "非常", "年", "用事"]
24 ["駐車場", "海外", "車", "外国人", "日本", "宿泊客", "日本人", "地下", "客", "駐車"]
25 ["出張", "コンビニ", "近い", "X分", "徒歩", "周辺", "JR", "地下鉄", "静か", "飲食店"]
26 ["大変", "イベント", "プラン", "家族", "JAL", "クーポン", "参加", "食事", "お正月", "心配"]
27 ["レベル", "期待", "評価", "高い", "無い", "接客", "印象", "悪い", "低い", "ハード"]
28 ["古い", "建物", "新しい", "リニューアル", "きれい", "池袋", "自体", "老舗", "設備", "清潔"]
29 ["渋谷", "場所", "汐留", "方面", "渋谷駅", "銀座", "ビル", "新橋駅", "X階", "近い"]
30 ["庭園", "庭", "結婚式", "素晴らしい", "日本庭園", "散歩", "椿山荘", "散策", "友人", "都内"]
```

上記の例だと、トピック1: 料理、トピック3: 眺望、トピック5: 料金、トピック6: 接客、といったトピックが抽出できているのが確認できる

## タグ付け

可視化したトピックの結果からタグを設定する
```julia
julia> tags = Dict(1 => "料理", 3 => "眺望", 5 => "料金", 6 => "接客", ...)
```

モデルパラメータをもとに口コミ文に対してタグ付けを行う
```julia
julia> include("Tagger.jl")
julia> using Main.Tagger
julia> Tagger.set_params(model_params, tags)
```

口コミ文のタグを推定する
```julia
julia> doc = "眺めが最高のホテルです。フロントの方も気持ちのよい対応でした。"
julia> Tagger.tagging(doc, ɛ=0.05)
"眺望"
"接客"
julia> Tagger.tagging(doc, ɛ=0.0, debug=true)
("眺望", 0.20695086589781417)
("料金", 0.00010087121042970607)
("接客", 0.17288492836449243)
("料理", 0.0024991764046796206)
```
