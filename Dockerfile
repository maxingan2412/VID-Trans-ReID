# 使用一个基础镜像，比如 Miniconda
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /work

# 复制 Conda 环境文件到容器中
#COPY . /work/

# 使用 Conda 创建一个新的环境，并安装环境中的所有依赖

RUN cd /work/VID-Trans-ReID && git clone https://github.com/maxingan2412/VID-Trans-ReID.git && conda create -n vid python=3.7 && \
    conda activate vid && pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r requirements.txt &&


# 设置要运行的命令或启动脚本
#CMD ["your_start_command_here"]   112313131313
