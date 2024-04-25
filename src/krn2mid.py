import os
import music21 as m21
from preprocess import MUSIC_DIR, DATASET_PATH, DURATIONS, acceptable_durations, transpose, encode


def krn2mid(music_path: str, step_duration=0.25):
    """
    指定一首歌krn转为midi
    """
    if music_path[-3:] != "krn":
        raise FileExistsError(f"{music_path} is not a 'krn' file.")
    
    music = m21.converter.parse(music_path)

    if not acceptable_durations(music, DURATIONS):
        print(f"{music_path} is not acceptable")
        return
    
    music = transpose(music)
    music = encode(music)
    music = music.split()

    stream = m21.stream.Stream()

    start_token = None
    step_counter = 1

    for idx, token in enumerate(music):
        # handle case in which we have a note/rest
        if token != "_" or idx + 1 == len(music):
            # ensure we're dealing with note/rest beyond the first one
            if start_token is not None:
                quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1
                # handle rest
                if start_token == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                # handle note
                else:
                    m21_event = m21.note.Note(int(start_token), quarterLength=quarter_length_duration)
                stream.append(m21_event)
                # reset the step counter
                step_counter = 1
            start_token = token

        # handle case in which we have a prolongation sign "_"
        else:
            step_counter += 1

    # write the m21 stream to a midi file
    save_dir = os.path.dirname(os.path.join(MUSIC_DIR % music_path.split("datasets/")[-1]))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, music_path.split("/")[-1].replace(".krn", ".mid"))
    stream.write("midi", save_dir)


def krn2mids(musics_path):
    """
    将一个目录下的所有krn转为midi
    """
    for path, subdirs, files in os.walk(musics_path):
        for file in files:
            if file[-3:] == "krn":
                music_path = os.path.join(path, file)
                krn2mid(music_path)


if __name__ == "__main__":
    krn2mids(DATASET_PATH)
