import os
import json
import gradio as gr
from datetime import datetime


WAV_DIR = r"path/to/wav_dir" # todo 这里改成你本地存放wav的目录
EMO_DICT = dict()
EMO_DIR = os.path.join(WAV_DIR, "emotion.json")
LOG_DIR = os.path.join(WAV_DIR, "label.log")
EMO_MAP = {
    "喜悦": "happy",
    "愤怒": "angry",
    "厌恶": "dislike",
    "低落": "depressed"
}


def walkdir(folder):
    for path, subdirs, files in os.walk(folder):
        for file in files:
            yield os.path.abspath(os.path.join(path, file))


MUSIC_PATH = []
for file in walkdir(WAV_DIR):
    if file[-3:] == "wav":
        MUSIC_PATH.append(file)


def get_next_path(path):
    index = MUSIC_PATH.index(path) + 1
    if index >= len(MUSIC_PATH):
        return "所有标注任务已经完成", MUSIC_PATH[0]
    return MUSIC_PATH[index], MUSIC_PATH[index]


def label_music(path: str, emo: str):
    EMO_DICT[path] = EMO_MAP[emo]
    with open(EMO_DIR, "w", encoding="utf-8") as f:
        json.dump(EMO_DICT, f, indent=4)
    time = datetime.now().strftime("%H:%M:%S")
    msg = f"The labeling was successful! \t\"{path.split('/')[-1]}\"\t is {emo}"
    print("\033[92m" + "[ OK ]" + "\033[0m\t" + msg)
    log = f"{time}\t{msg}"
    with open(LOG_DIR, "a", encoding="utf-8") as f:
        f.write(log + "\n")
    return log

with gr.Blocks() as demo:
    with gr.Row():
        path = gr.Textbox(value=MUSIC_PATH[0], scale=3, label="音乐地址")
        refresh = gr.Button("Refresh", scale=1)
    music = gr.Audio(value=MUSIC_PATH[0])
    refresh.click(fn=get_next_path, inputs=path, outputs=[path, music])

    label = gr.Radio(choices=list(EMO_MAP.keys()), label="音乐情感")
    remind = gr.Textbox(value="请开始标注情感")

    submit = gr.Button("Submit")
    submit.click(fn=label_music, inputs=[path, label], outputs=[remind])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=51234)
