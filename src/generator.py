import os
import json
import random
import argparse
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from preprocess import SAVE_MODEL_PATH, SEQUENCE_LENGTH, MAP_PATH, MUSIC_DIR, one_hot_encode

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEEDS = [
    "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _",
    "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
]

parser = argparse.ArgumentParser(description="Generator Music from Emotions.")
parser.add_argument("-m", "--model_path", type=str, default=SAVE_MODEL_PATH, help="Directory for the model.")
parser.add_argument("-s", "--seed", type=str, default=random.choice(SEEDS), help="Random seed for the music.")
parser.add_argument("-t", "--temperature", type=float, default=0.3, help="Mode temperature.")
args = parser.parse_args()

model_path = args.model_path
seed = args.seed
temperature = args.temperature


class Generator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAP_PATH, "r") as fp:
            self.music_map = json.load(fp)

        self.start_tokens = ["/"] * SEQUENCE_LENGTH


    def generate_music(self, seed, num_steps, max_sequence_length, temperature):
        # create seed with start tokens
        seed = seed.split()
        music = seed
        seed = self.start_tokens + seed

        # map seed to int
        seed = [self.music_map[token] for token in seed]

        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            one_hot_seed = one_hot_encode(seed, num_classes=len(self.music_map))
            # (1, max_sequence_length, num of tokens in the vocabulary)
            one_hot_seed = np.expand_dims(one_hot_seed, axis=0)

            # make a prediction
            probabilities = self.model.predict(one_hot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self.sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_token = [k for k, v in self.music_map.items() if v == output_int][0]

            # 检查是否停止
            if output_token == "/":
                break

            # update music
            music.append(output_token)

        return music


    def sample_with_temperature(self, probabilites, temperature):
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))
        index = np.random.choice(choices, p=probabilites)
        return index


    def save_music(self, music, step_duration=0.25):
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
        save_dir = os.path.dirname(os.path.join(MUSIC_DIR % "generator/"))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, "migic.mid")
        stream.write("midi", save_dir)


if __name__ == "__main__":
    generator = Generator(model_path=model_path)
    music = generator.generate_music(seed, 500, SEQUENCE_LENGTH, temperature)
    generator.save_music(music)
    print("The magic has happened.")
