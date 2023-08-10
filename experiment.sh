#!/bin/bash

# 获取当前日期和时间
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 设置日志文件路径
LOG_FILE="jilulog/experiment_$CURRENT_TIME.txt"

# 检查并创建日志文件目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 记录开始时间和命令
echo "Experiment started at $CURRENT_TIME" >> "$LOG_FILE"
echo "Command: $*" >> "$LOG_FILE"

# 执行命令并将输出追加到日志文件，同时输出到终端（保留控制字符）
"$@" 2>&1 | tee -ai "$LOG_FILE"

# 记录结束时间
echo "Experiment ended at $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
