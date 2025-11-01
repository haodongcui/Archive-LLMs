

## 环境配置
```
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# 创建和激活环境
conda create -n llm-torch2.8-cu12.9-py3.10 python=3.10 -y
conda activate llm-torch2.8-cu12.9-py3.10

# 包下载
pip install "D:\Backup\whl\torch-2.8.0+cu129-cp310-cp310-win_amd64.whl"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install jupyter notebook
pip install tiktoken

pip install matplotlib
pip install pandas
pip install tqdm

```

## 相关资料
- [LLM张老师(LLM从零到一)](https://space.bilibili.com/3546611527453161/lists/2386239?type=season)
- [Transformer Architecture (LLMs: Zero-to-Hero) | by Wayland Zhang | Medium](https://medium.com/@waylandzhang/transformer-architecture-llms-zero-to-hero-98b1ee51a838)
- [B站课程-浙大LLMs](https://www.bilibili.com/video/BV1bynBzbEEd?spm_id_from=333.788.videopod.episodes&vd_source=acbdb8fe72197a35f3556cf0747ef4e4&p=4)
- [翻译任务](https://github.com/Fugtemypt123/mytransformer/blob/master/train.py)




## 注意

### 索引后的维度
- x -> torch.Size([4, 4, 16, 16])
- x[:,0,:,:] -> torch.Size([4, 16, 16]) 被索引的维度直接删掉了

### 使用 .to(device) 的方法
有两种选择，
- 一种是在定义class的内部添加.to(device)
- 一种是在实例化的时候添加.to(device)
推荐第二种，原因一是 class 定义时可能会嵌套多层，实例化时可以一次性添加所有变量，原因二是可以方便的管理 device

