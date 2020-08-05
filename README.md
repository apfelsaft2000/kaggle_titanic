# kaggle：titanicに挑戦
CNNに全部ぶっ込めばそこそこのスコアを達成できる説の検証

## 実行環境
- [Anaconda3](https://www.anaconda.com/distribution/)
- Python3.6
- pytorch
- tqdm
- pandas

## 手順
### 1.環境構築
シェルスクリプトを実行し、環境構築する

`sh requirements.sh`

### 2.前処理
cabin,Ticketなどが数字じゃないかつ生存に関係しているのか分析しないとわからないのでとりあえずスルーする。
前処理では以下を行う
- 性別,Embarkedなどは0,1などの数字に置換
- cabin,Ticket,nameを削除
- Ageの欠損値をAgeの平均値で補完
- 全体のデータを0~1で正規化する 
- テストデータに正解ラベルを付与する
- 訓練データの正解ラベルを一番右側に移動

上記のプロセスは以下のスクリプトを実行することで作成

`python preprcessing.py`

### 3.訓練とテスト
`python train.py`

### 4.結果
 ひとまず、コンソールで確認
