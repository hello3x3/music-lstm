import os
import json
import numpy as np
from tqdm import tqdm
import music21 as m21

os.chdir(os.path.dirname(os.path.realpath(__file__)))

with open("../config.json", "r", encoding="utf-8") as f:
    cfg: dict = json.load(f)

DATASET_PATH = os.path.abspath(os.path.join("../", cfg["DATASET_PATH"]))
SAVE_DIR = os.path.join(os.path.abspath(os.path.join("../", cfg["SAVE_DIR"])), DATASET_PATH.split("/")[-1])
SEQUENCE_LENGTH = cfg["SEQUENCE_LENGTH"]
PREPROCESS_DATASET_DIR = os.path.join(SAVE_DIR, "temp")
MAP_PATH = os.path.join(SAVE_DIR, "music_map.json")

DURATIONS = [
    0.25,   # 十六分之一音符
    0.5,    # 八分之一音符
    0.75,
    1.0,    # 四分之一音符
    1.5,
    2,      # 半音符
    3,
    4       # 全音符
]


def load_musics(dataset_path: str):
    """
    加载kern格式的音乐片段
    """
    musics = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                music = m21.converter.parse(os.path.join(path, file))
                musics.append(music)
    return musics


def acceptable_durations(music, durations):
    """
    判断音乐是否合格
    """
    for note in music.flat.notesAndRests:
        if note.duration.quarterLength not in durations:
            return False
    return True


def transpose(music):
    """
    将音乐转到 C maj / A min
    """

    # get key from the song
    parts = music.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = music.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_music = music.transpose(interval)
    return tranposed_music


def encode(music, time_step=0.25):
    """
    将乐谱转换为类似时间序列的音乐表示法。编码列表中的每一项都代表 “min_duration ”四分音符长度。每一步使用的符号是：整数表示 MIDI 音符，“r ”表示休止符，“_”表示带入新时间步的音符/休止符。下面是一个编码示例：

    ["r", "_", "60", "_", "_", "_", "72" "_"]
    """

    encoded_music = []

    for event in music.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_music.append(symbol)
            else:
                encoded_music.append("_")

    encoded_music = " ".join(map(str, encoded_music))
    return encoded_music


def preprocess(dataset_path: str):
    # load folk songs
    print("Loading songs...")
    musics = load_musics(dataset_path)
    print(f"Loaded {len(musics)} songs.")

    if not os.path.exists(PREPROCESS_DATASET_DIR):
        os.makedirs(PREPROCESS_DATASET_DIR)

    for idx, music in enumerate(tqdm(musics)):
        # filter out songs that have non-acceptable durations
        if not acceptable_durations(music, DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        music = transpose(music)

        # encode songs with music time series representation
        encoded_music = encode(music)

        # save songs to text file
        save_path = os.path.join(PREPROCESS_DATASET_DIR, f"music{idx}.txt")
        with open(save_path, "w") as f:
            f.write(encoded_music)


def load(file_path):
    with open(file_path, "r") as f:
        music = f.read()
    return music


def create_musics_seq(preprocess_dataset_path, sequence_length):
    """
    生成一个文件，整理所有编码歌曲并生成全部的音乐序列。
    """

    new_music_delimiter = "/ " * sequence_length
    musics = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(preprocess_dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            music = load(file_path)
            musics = musics + music + " " + new_music_delimiter

    # remove empty space from last character of string
    musics = musics[:-1]

    # save string that contains all the dataset
    save_dir = os.path.join(SAVE_DIR, "musics_seq.txt")
    with open(save_dir, "w") as f:
        f.write(musics)
    return musics


def create_map(musics, map_path):
    music_map = dict()

    # identify the vocabulary
    musics = musics.split()
    vocab = list(set(musics))

    # create mappings
    for idx, symbol in enumerate(vocab):
        music_map[symbol] = idx

    # save voabulary to a json file
    with open(map_path, "w") as f:
        json.dump(music_map, f, indent=4)


def convert_musics_to_int(musics):
    int_musics = []

    # load mappings
    with open(MAP_PATH, "r") as fp:
        music_map = json.load(fp)

    # transform songs string to list
    musics = musics.split()

    # map songs to int
    for symbol in musics:
        int_musics.append(music_map[symbol])

    return int_musics

def one_hot_encode(inputs, num_class=None):
    inputs = np.array(inputs)
    inputs_shape = inputs.shape

    if inputs_shape and inputs_shape[-1] == 1 and len(inputs_shape) > 1:
        inputs_shape = tuple(inputs_shape[:-1])
    
    inputs = inputs.ravel()
    
    if not num_class:
        num_class = np.max(inputs) + 1
    n = inputs.shape[0]

    ctg = np.zeros((n, num_class), dtype="float32")
    ctg[np.arange(n), inputs] = 1

    output_shape = inputs_shape + (num_class, )
    ctg = np.reshape(ctg, output_shape)
    return ctg


def generate_training_sequences(sequence_length):
    """
    创建用于训练的输入和输出数据样本。每个样本都是一个序列。
    """

    # load songs and map them to int
    musics = load(os.path.join(SAVE_DIR, "musics_seq.txt"))
    int_musics = convert_musics_to_int(musics)

    inputs = []
    targets = []

    # 生成基于sequence_length时间窗口的训练序列
    num_sequences = len(int_musics) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_musics[i:i+sequence_length])
        targets.append(int_musics[i+sequence_length])

    # one-hot encode the sequences
    vocab_size = len(set(int_musics))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = one_hot_encode(inputs, num_class=vocab_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


if __name__ == "__main__":
    preprocess(DATASET_PATH)
    musics = create_musics_seq(PREPROCESS_DATASET_DIR, SEQUENCE_LENGTH)
    create_map(musics, MAP_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
