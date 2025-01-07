#!/bin/bash
# 下载数据集
huggingface-cli download --repo-type dataset Skylion007/openwebtext --local-dir openwebtext

cd openwebtext/subset
# 解压每一个子集
for subset in *.tar; do
  echo Processing $subset
  tar -xf $subset
  cd openwebtext
  for datafile in *.xz; do
    tar -xf $datafile
    # 解压出文本文件并合并成一个文件
    cat *.txt >> ../../openwebtext.txt
    rm *.txt
    rm $datafile
  done
  cd ..
done
