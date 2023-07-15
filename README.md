# 爆速🚀MEMD \(bakusoku-memd\)

[//]: # (image or gif)

## Overview

Multivariate Empirical Mode Decomposition (MEMD)を 🚅爆速💨 で計算するプログラム  
ライブラリ版もあるよ

## Getting Started

1. `docker-compose run --rm bakusoku-memd`
2. `mpiexec -n 2 ./bakusoku-memd ./sampledata/ndarray_173_5_seed0.csv out_imf 32`
3. MEMDの結果が `./out_imfXXX` に出力される

## Features

- 多次元の時系列データのCSVファイルを読み込み、 MEMDによって複数のIMF、及び残差を複数のCSVとして出力する
    - 出力ファイル名のプレフィックスと方向ベクトルの次元数を設定できるよ

## Requirement

- Docker \(かんたん👌\)

あるいは

- C++17が使える新しめなコンパイラ
- CMake 3.16以上
- OpenBLAS
- MPI実行環境

## Usage

- 読み込むデータファイル、imfを書き出すファイルの接頭辞、単位ベクトルの次元数を指定して実行

  `./bakusoku-memd ./path/to/sampledata/ndarray_173_5_seed0.csv out_imf 32`
- 計算精度(pythonのMEMDライブラリを基準とする)を確認するテスト (詳しくはテストのソースコードを読んでね)

  `./bakusoku-memd-lib-test`
- MPIでプロセス数を指定して実行する

  `mpiexec -n 2 ./bakusoku-mend...`

## Installation

- Dockerを使う場合 \(おすすめ 🎉️\)
    1. `docker-compose run --rm bakusoku-memd`
        - これだけでおk 👍
- minicondaを使う場合
    1. minicondaをインストール
       `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`
    2. Requirementのツールをconda環境にインストール
        1. `conda config --add channels conda-forge`
        2. `conda install cxx-compiler cmake openblas openmpi`
    3. ビルド
        1. `cd bakusoku-memd && mkdir build && cd ./build`
        2. `cmake -DCMAKE_BUILD_TYPE=Release ..`
        3. `make`
        4. `build/bakusoku-memd-*` ディレクトリに各種実行ファイル、テストが生成される
- それ以外の場合
    1. がんばりましょう。\( ´ ▽ \` \)ﾉ

もし必須ライブラリをパッケージマネージャを使わずインストールした場合、 `bakusoku-memd-lib/local-properties.cmake`を作成しよう!  
詳細は`local-properties-default.cmake`を読んでね

[//]: # (## Reference)

## Author

生野・董研究室 島田雄気  
[研究室](https://wiki.ikulab.org/)

## Licence

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](./LICENSE.md)

[//]: # (&#40;https://creativecommons.org/licenses/by-nc-sa/4.0/&#41;)