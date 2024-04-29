# music-lstm

使用lstm生成音乐

## Requirements

```bash
conda create -n music-lstm python=3.9 -y

pip install -r requirements.txt
```

## Quick Start

### Preprocess

1. 修改config.json里的路径

```json
{
    "DATASET_PATH": "./datasets/europa/deutschl/erk",   // 数据集路径，预处理时会遍历每一个子文件
    "SAVE_DIR": "./preprocessed_datasets",              // 预处理后的数据存放文件夹
    "MUSIC_DIR": "./musics",                            // 生成的音乐或由krn格式转化成的音乐路径
    "CKPT_DIR": "./ckpt",                               //预训练模型的存储路径
    "SEQUENCE_LENGTH": 64                               //序列长度
}
```

2. 对数据集中的krn格式音乐进行预处理

```bash
python src/preprocess.py
```

3. 将krn格式的音乐转换成midi格式，`src/krn2mid.py`中提供两个函数：

`krn2mid`：指定一首歌krn转为midi；

`krn2mids`：将一个目录下的所有krn转为midi；

```bash
python src/krn2mid.py
```

4. 对音乐进行情感标注：

注意：前提必须是已经得到wav格式的音乐，目录为`musics/wav/...`，标注完保存的json文件在同级目录中的`emotion.json`。

默认网页端口为51234

```bash
python src/label.py
```

windows下可以参考`src/label_for_win.py`，注意，需要改该文件中的`WAV_DIR`.

### Train

本项目依赖`tensorflow-gpu==2.7.0`和`protobuf==3.19.0`，训练时大约占用显存40GB

```bash
python src/train.py
```

默认训练50个epochs，在A6000-48GB的GPU下，一个epoch大约需要72s，总共用时30分钟，模型将保存在`CKPT_DIR`下。

### Generator

程序调用预训练好的模型权重文件，根据特定的种子序列生成音乐，保存在`MUSIC_DIR`下。

```bash
python src/generator.py
```

### Other

将midi转成mp3格式时，需要用到工具`fluidsynth`，这里提供一个最小的docker实现批量格式转换。

```bash
# 1. 从源码build开始
# build
docker build -t music-lstm:v0.1 .

# run
docker run -it -v /data/shujiuhe/music-lstm/musics:/root/musics music-lstm:v0.1

# 2. 从外部导入
# import
docker import music-lstm.tar music-lstm:v0.1

# run
docker run -it -v /data/shujiuhe/music-lstm/musics:/root/musics music-lstm:v0.1
```

批量转化的代码是`src/mid2wav.py`，里面同样提供两个函数：

`mid2wav`：指定一首歌midi转成wav；

`mid2wavs`：将一个目录下的所有midi转为wav；

```bash
python3 /root/mid2wav.py
```
