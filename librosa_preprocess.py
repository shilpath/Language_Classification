import librosa
import os
import numpy as np


def audio_preprocess():
    #Set the path of your folder where the audio files are saved.
    folder_name = "./train/"

    english_audio = []
    folder1 = folder_name + "train_english/"
    print("Processing english files")
    count = 0
    files = sorted(os.listdir(folder1))
    for file in files:
        #if count<1:      #If you want to load only the first n number of files.
            print(file)
            y, sr = librosa.load(folder1+file, sr=16000)
            intervals = librosa.effects.split(y=y, top_db=40)     #To remove silence from audio file. Returns non-silent intervals
            new_y = y[intervals[0][0]:intervals[0][1]]
            for item in intervals[1:]:
                new_y = np.concatenate([new_y, y[item[0]:item[1]]])
            mat = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
            english_audio.append(mat)
            #count+=1

    hindi_audio = []
    folder1 = folder_name + "train_hindi/"
    count = 0
    print("Processing hindi files")
    files = sorted(os.listdir(folder1))
    for file in files:
        #if count<1:
            print(file)
            y, sr = librosa.load(folder1+file, sr=16000)
            intervals = librosa.effects.split(y=y, top_db=40)
            new_y = y[intervals[0][0]:intervals[0][1]]
            for item in intervals[1:]:
                new_y = np.concatenate([new_y, y[item[0]:item[1]]])
            mat = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
            hindi_audio.append(mat)
            #count+=1

    mandarin_audio = []
    folder1 = folder_name + "train_mandarin/"
    count = 0
    print("Processing mandarin files")
    files = sorted(os.listdir(folder1))
    for file in files:
        #if count<1:
            print(file)
            y, sr = librosa.load(folder1+file, sr=16000)
            intervals = librosa.effects.split(y=y, top_db=40)
            new_y = y[intervals[0][0]:intervals[0][1]]
            for item in intervals[1:]:
                new_y = np.concatenate([new_y, y[item[0]:item[1]]])
            mat = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
            mandarin_audio.append(mat)
            #count+=1

    return english_audio, hindi_audio, mandarin_audio
