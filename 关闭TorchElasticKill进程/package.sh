# 遍历当前一级子目录，逐个打包（解压无外层目录）
for dir in */; do
  name="${dir%/}"
  tar -zcvf "${name}.tar.gz" -C "$dir" .
done
