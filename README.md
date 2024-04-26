# music-lstm

使用lstm生成音乐

## requirements

```bash
conda create -n music-lstm python=3.9 -y

pip install -r requirements.txt
```

## quick start

### preprocess

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

3. 将krn格式的音乐转换成midi格式

```bash
python src/krn2mid.py
```

### train

本项目依赖`tensorflow-gpu==2.7.0`和`protobuf==3.19.0`，训练时大约占用显存40GB

```bash
python src/train.py
```

默认训练50个epochs，在A6000-48GB的GPU下，一个epoch大约需要72s，总共用时30分钟，模型将保存在`CKPT_DIR`下。

### generator

程序调用预训练好的模型权重文件，根据特定的种子序列生成音乐，保存在`MUSIC_DIR`下。

```bash
python src/generator.py
```
