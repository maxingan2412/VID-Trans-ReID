#!/bin/bash
#SBATCH --job-name=vit32
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=out/diff_train-%j.out
#SBATCH --error=out/diff_train-%j.err

# 获取当前日期和时间
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 获取当前时间的秒数
START_SECONDS=$(date +%s)

# 设置日志文件路径
LOG_FILE="jilulog/experiment_$CURRENT_TIME.txt"

# 检查并创建日志文件目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 记录开始时间
echo "Experiment started at $CURRENT_TIME" >> "$LOG_FILE"

echo "setting : bs 32 epoch 240" >> "$LOG_FILE"
# 执行的命令，并把它们的输出追加到日志文件

nvidia-smi >> "$LOG_FILE" 2>&1

conda env list >> "$LOG_FILE" 2>&1

COMMAND="/home/ma1/anaconda3/envs/vid/bin/python -u VID_Trans_ReID.py --Dataset_name 'Mars' --ViT_path 'jx_vit_base_p16_224-80ecf9dd.pth' --epochs 240"
echo "Executing: $COMMAND" >> "$LOG_FILE"
eval "$COMMAND" >> "$LOG_FILE" 2>&1

# 记录结束时间
echo "Experiment ended at $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
