#!/bin/bash
set -e
[ -z "$1" ] && { echo "用法：$0 使用者@來源IP"; exit 1; }
mkdir -p "$HOME/models/qwen38"
rsync -avP "$1:~/models/qwen38/" "$HOME/models/qwen38/"
