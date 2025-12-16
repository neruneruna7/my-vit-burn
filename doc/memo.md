# vitの構造
自然言語のトランスフォーマーを意識している．
BERTを学んでからのほうがやりやすい？

次の5つの段階があるらしい
1. 画像パッチ化
2. CLSトークン埋め込み
3. トランスフォーマー入力
4. MLP入力

### 画像パッチ化
画像を任意の数に等分.
それぞれを単語とみなす．

んでむりやりトランスフォーマーに入力するらしい．
全結合層に入れて，トランスフォーマーにねじ込めるようにする．

### CLSトークン埋め込み
CLSというのはクラスの略らしい．
画像全体の特徴量を集める．

すると，クラスの出力のみを使って画像分類できるとのこと．

学習可能な変数をCLSトークンとして定義．
これをベクトル化されたパッチ列の先頭に配置．

これなしでもVITは機能するらしい．clsトークン以外をmlpにぶち込めば．
NLP用のトランスフォーマーをそのまま使えるようにするための工夫らしい．

#### ポジション埋め込み
トランスフォーマーには入力順という概念がない．
ので，入力と画像位置の対応関係を明示する．
1次元の学習可能な変数を加算すればいいらしい．


### トランスフォーマー入力
NLP用のトランスフォーマーをそのまま使う．

### MLP入力
10次元出力のMLPを作って，トランスフォーマー最終層のCLSトークン出力をぶち込む．

# テンソルの形状確認
```
=== Forward Pass ===
Input Shape: torch.Size([128, 3, 32, 32])
After Chunk Shape: [torch.Size([128, 3, 8, 32]), torch.Size([128, 3, 8, 32]), torch.Size([128, 3, 8, 32]), torch.Size([128, 3, 8, 32])]
After Stack Shape: torch.Size([128, 4, 3, 8, 32])
After Column Chunk Shape: [torch.Size([128, 4, 3, 8, 8]), torch.Size([128, 4, 3, 8, 8]), torch.Size([128, 4, 3, 8, 8]), torch.Size([128, 4, 3, 8, 8])]
After Concat Patches Shape: torch.Size([128, 16, 3, 8, 8])
After Patchify Shape: torch.Size([128, 16, 3, 8, 8])
After Flatten Shape: torch.Size([128, 16, 192])
After Patch Embedding Shape: torch.Size([128, 16, 96])
After Transformer Encoder Shape: torch.Size([128, 17, 96])
Output Shape: torch.Size([128, 10])
=====================
```

最初は4次元
Rowをチャンク化して5次元に
それをStackして4次元に
Colをチャンク化して5次元に
それをConcatして5次元に
パッチ処理後は5次元
それをFlattenして3次元に
それをパッチ埋め込みして3次元に
トランスフォーマー通過後は3次元
MLP通過後は2次元

# BIB
- https://qiita.com/nknknaoto/items/615e8057db0a45d7b1be
- 