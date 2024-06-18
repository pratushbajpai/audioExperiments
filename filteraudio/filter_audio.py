"""
Filter Audio

This Python script uses speech recognition and audio manipulation
to remove fillter words in an audio file.
It transcribes the audio, searches for noise words as identified by whisper, 
and splits the file based on the timestamps in those words in multiple segments. 
It later merges all the segments without noise and merge them in a new single file, 
free of filler words.
"""

import argparse
import csv
import sys
import os
#import whisper
import whisper_timestamped as whisper
#from stable_whisper import modify_model
from pydub import AudioSegment

def parse_tsv(tsv_string):
    """
    Parses a TSV string into a list of tuples.
    """
    words = []
    reader = csv.reader(tsv_string.split('\n'), delimiter='\t')
    for row in reader:
        if len(row) == 3:
            start_time, end_time, word = row
            words.append((float(start_time), float(end_time), word))
    return words

def convert_to_wav(input_file):
    """
    Converts the input audio file to WAV format.
    """
    audio = AudioSegment.from_file(input_file)
    wav_file = os.path.splitext(input_file)[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

def transcribe_audio(audio_file):
    """
    Transcribes the audio using the Whisper ASR model.
    """
    model = whisper.load_model("tiny", device="cpu")
    #modify_model(model)
    audio = whisper.load_audio(audio_file)
    #result = model.transcribe(audio_file, language="en", detect_disfluencies=True, suppress_silence=True, no_speech_threshold=0.2, word_timestamps=True)
    result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True,beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    print(result)
    return result
    #return result.to_tsv('', segment_level=True, word_level=True)

def find_filler_words(transcribed_results):
    filler_words=[]
    for text in transcribed_results["segments"]:
        for word in text["words"]:
            if "[*]" in word["text"]:
                filler_words.append([word["start"], word["end"]])
    return filler_words

def find_split_timestamps(transcribed_results, filler_words):
    split_times = []
    final_end_time = transcribed_results["segments"][-1]["end"]

    for i in range(len(filler_words)):
        if i == 0:
            start = 0
            end = filler_words[i][0]
        else:
            start = filler_words[i-1][1]
            end = filler_words[i][0]  
            filler_words[i]
        split_times.append([start, end])
    split_times.append([filler_words[-1][1], final_end_time])
    return split_times

def split_audio_by_timestamps(audio_file, split_timestamps):
    audio_files = []
    audio = AudioSegment.from_file(audio_file)
    for i, (start_time, end_time) in enumerate(split_timestamps):
        segment = audio[start_time * 1000:end_time * 1000]
        segment_file = f"segment_{i}.wav"
        segment.export(segment_file, format="wav")
        audio_files.append(segment_file)
    return audio_files

def merge_audio_segments(audio_files):
    audio_segments = [AudioSegment.from_file(file) for file in audio_files]
    merged_audio = sum(audio_segments)

    # remove intermittent files
    for file in audio_files:
        remove_file(file)

    return merged_audio

def remove_file(file_path):
    """
    Removes the specified file from the filesystem.
    """
    # pylint: disable=broad-exception-caught
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")

def main():
    """
    Main function to handle command line arguments and execute the profanity removal process.
    """
    # parser = argparse.ArgumentParser(description='Profanity Silencer for Audio Files')
    # parser.add_argument('input', help='Input audio file path')
    # parser.add_argument('output', help='Output audio file path')
    # args = parser.parse_args()
    input_file = None
    output_file = None

    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = 'test.mp3'
        output_file = 'output.mp3'
    
    wav_file = ''

    if not input_file.lower().endswith(".wav"):
        print("Converting input audio to WAV format...")
        wav_file = convert_to_wav(input_file)
        input_file = wav_file
# pylint: disable=broad-exception-caught
    try:
        #transcribe audio to clean definiciencies by whisper
        transcribed_results = transcribe_audio(input_file)
        
        #find filler words timestamp
        filler_word_timestamps = find_filler_words(transcribed_results)

        #find split time stamps for the trsncribed audio to remove fillers
        split_timestamps = find_split_timestamps(transcribed_results, filler_word_timestamps) 

        #break the audio into segments of non-filler words
        audio_segments = split_audio_by_timestamps(input_file, split_timestamps) 

        # merge all the audio into one file
        merged_audio = merge_audio_segments(audio_segments)
        
        if not output_file.lower().endswith(".wav"):
            print("Converting output audio to mp3 format...")
            merged_audio.export(output_file, format="mp3")
        else:
            merged_audio.export(output_file, format="wav")

        print("audio filterd successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if wav_file:
            remove_file(wav_file)

if __name__ == "__main__":
    main()
