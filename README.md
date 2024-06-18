Repo for audio experiments.
filteraudio - proprtypes filtering noise from audio input and generates a new file without any filler words. This project leverages https://github.com/linto-ai/whisper-timestamped to parse timestamps for noise words. It then splits the filtered audio segments based on non-noise segments and merges them in a new file which is free of filler words.
