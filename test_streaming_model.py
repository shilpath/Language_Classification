import numpy as np
import tensorflow
import librosa
import os

audio = []

##replace folder1 path with the test audio files path
folder1 = "./train/train_english/"
count = 0
for file in os.listdir(folder1):
    #if count<1:
        y, sr = librosa.load(folder1+'/'+file, sr=16000)
        intervals = librosa.effects.split(y=y, top_db=40)
        new_y = y[intervals[0][0]:intervals[0][1]]
        for item in intervals[1:]:
            new_y = np.concatenate([new_y, y[item[0]:item[1]]])
        mat = librosa.feature.mfcc(y=new_y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        audio.append(mat)
        #count+=1

audio_concat = audio[0]
for i in range(1, len(audio)):
    audio_concat = np.concatenate((audio_concat, audio[i]), axis=1)

train_seq_len = 400
feature_dim = 64

concat_ind = int(audio_concat.shape[1]/train_seq_len) * train_seq_len
audio_concat = np.array(audio_concat[:, :concat_ind])
n_eng = int(audio_concat.shape[1]/train_seq_len)
audio_concat = np.reshape(audio_concat.T, (n_eng, train_seq_len, feature_dim))
print(audio_concat.shape)

x_test_data = audio_concat

streaming_model = tensorflow.keras.models.load_model('streaming_model.hdf5')

print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
for s in range(x_test_data.shape[0]):
    print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
    for n in range(train_seq_len):
        in_feature_vector = x_test_data[s][n].reshape(1, 1, feature_dim)
        single_pred = streaming_model.predict(in_feature_vector)[0]
        print(' Streaming-Model Prediction [English, Hindi, Mandarin]: ', single_pred)
    streaming_model.reset_states()

