import os
import subprocess
from tqdm import tqdm

# 注意此文件中的目录关系都是docker中的
MID_PATH = "/root/musics/mid"
WAV_PATH = "/root/musics/wav"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# print(os.path.abspath(__file__))

def mid2wav(music_path: str):
    """
    指定一首歌midi转成wav
    """
    music_name = music_path.split("/")[-1].replace(".mid", ".wav")
    music_dir = os.path.dirname(music_path.split('.mid')[-2]).split("musics/")[-1].replace("mid/", "")

    save_dir = os.path.join(WAV_PATH, music_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, music_name)

    subprocess.call(
        [
            "fluidsynth", "-ni", "/root/Arachno.sf2", f"{music_path}", "-F", f"{save_dir}"
        ],
        stdout=subprocess.DEVNULL
    )


def mid2wavs(musics_path: str):
    """
    将一个目录下的所有midi转为wav
    """
    def walkdir(folder):
        for path, subdirs, files in os.walk(folder):
            for file in files:
                yield os.path.abspath(os.path.join(path, file))

    cnt = 0
    for _ in walkdir(musics_path):
        cnt += 1

    for music_path in tqdm(walkdir(musics_path), total=cnt):
        if music_path[-3:] == "mid":
            mid2wav(music_path)


if __name__ == "__main__":
    mid2wavs(MID_PATH)
