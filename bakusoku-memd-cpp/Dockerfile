# なるべく各キャッシュサイズを小さくするように書く
# 公式のDebianやUbuntuはapt clean が自動で実行される設定なのでわざわざ書かなくとも良い

ARG dir_workspace=bakusoku-memd
ARG dir_release=bakusoku-memd-release


FROM ubuntu:lunar-20230615 AS builder

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list \
    && apt update \
    && apt install -y g++ ninja-build cmake libopenblas-dev \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

ARG dir_workspace
ARG dir_release
ARG dir_build=cmake-build-release

COPY . /$dir_workspace

RUN cd $dir_workspace/ \
    && mkdir $dir_build \
    && cd ./$dir_build/ \
    && cmake -DCMAKE_BUILD_TYPE=Release -GNinja .. \
    && ninja \
    && mkdir ../../$dir_release \
    && mv ./bakusoku-memd-cpp/bakusoku-memd ../../$dir_release/ \
    && mv ../sampledata ../../$dir_release/ \
    && mv ./bakusoku-memd-lib/bakusoku-memd-lib-test ../../$dir_release/ \
    && mv ./bakusoku-memd-lib/res ../../$dir_release/ \
    && cd / \
    && rm -r $dir_workspace \
    && apt purge -y libopenblas-dev cmake ninja-build g++ --autoremove


FROM ubuntu:lunar-20230615
LABEL authors="Yuki"

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list \
    && apt update \
    && apt install libopenblas0-pthread -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

ARG dir_release

COPY --from=builder /$dir_release /$dir_release
WORKDIR /$dir_release

CMD ["/bin/bash"]