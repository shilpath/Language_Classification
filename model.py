from librosa_preprocess import audio_preprocess
import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import plot_model


def create_weights_matrix(labels, weights_mapping):
    # If using "return_sequences = True"
    if labels.ndim == 3:
        weights_matrix = np.zeros(labels.shape[0:2])

        for i,sample in enumerate(labels):
            for j,elem in enumerate(sample):
                weights_matrix[i,j] = weights_mapping[elem[0]]

    else:
        weights_matrix = np.zeros(labels.shape[0])
        for i,sample in enumerate(labels):
            weights_matrix[i] = weights_mapping[sample]

    return weights_matrix


np.random.seed(5)
english_audio, hindi_audio, mandarin_audio = audio_preprocess()
english_length = len(english_audio)
hindi_length = len(hindi_audio)
mandarin_length = len(mandarin_audio)

english_concat = english_audio[0]
for i in range(1, len(english_audio)):
    english_concat = np.concatenate((english_concat, english_audio[i]), axis=1)

hindi_concat = hindi_audio[0]
for i in range(1, len(hindi_audio)):
    hindi_concat = np.concatenate((hindi_concat, hindi_audio[i]), axis=1)

mandarin_concat = mandarin_audio[0]
for i in range(1, len(mandarin_audio)):
    mandarin_concat = np.concatenate((mandarin_concat, mandarin_audio[i]), axis=1)

train_seq_len = 400
feature_dim = 64

english_concat_ind = int(english_concat.shape[1]/train_seq_len) * train_seq_len
english_concat = np.array(english_concat[:, :english_concat_ind])
n_eng = int(english_concat.shape[1]/train_seq_len)
english_concat = np.reshape(english_concat.T, (n_eng, train_seq_len, feature_dim))
english_op = np.zeros((n_eng, train_seq_len, 1))
print(english_concat.shape)

hindi_concat_ind = int(hindi_concat.shape[1]/train_seq_len) * train_seq_len
hindi_concat = np.array(hindi_concat[:, :hindi_concat_ind])
n_hindi = int(hindi_concat.shape[1]/train_seq_len)
hindi_concat = np.reshape(hindi_concat.T, (n_hindi, train_seq_len, feature_dim))
hindi_op = np.ones((n_hindi, train_seq_len, 1))
print(hindi_concat.shape)

mandarin_concat_ind = int(mandarin_concat.shape[1]/train_seq_len) * train_seq_len
mandarin_concat = np.array(mandarin_concat[:, :mandarin_concat_ind])
n_mandarin = int(mandarin_concat.shape[1]/train_seq_len)
mandarin_concat = np.reshape(mandarin_concat.T, (n_mandarin, train_seq_len, feature_dim))
mandarin_op = np.ones((n_mandarin, train_seq_len, 1))*2
print(mandarin_concat.shape)

#-----------------------------------------------------------------------
eng_val_len = int(np.floor(0.15*english_concat.shape[0]))
english_val = english_concat[:eng_val_len, :, :]
english_op_val = english_op[:eng_val_len, :, :]

english_train = english_concat[eng_val_len:, :, :]
english_op_train = english_op[eng_val_len:, :, :]
#-----------------------------------------------------------------------

hindi_val_len = int(np.floor(0.15*hindi_concat.shape[0]))
hindi_val = hindi_concat[:hindi_val_len, :, :]
hindi_op_val = hindi_op[:hindi_val_len, :, :]

hindi_train = hindi_concat[hindi_val_len:, :, :]
hindi_op_train = hindi_op[hindi_val_len:, :, :]
#-----------------------------------------------------------------------

mandarin_val_len = int(np.floor(0.15*mandarin_concat.shape[0]))
mandarin_val = mandarin_concat[:mandarin_val_len, :, :]
mandarin_op_val = mandarin_op[:mandarin_val_len, :, :]

mandarin_train = mandarin_concat[mandarin_val_len:, :, :]
mandarin_op_train = mandarin_op[mandarin_val_len:, :, :]
#-----------------------------------------------------------------------

print(english_val.shape, english_train.shape)
print(hindi_val.shape, hindi_train.shape)
print(mandarin_val.shape, mandarin_train.shape)

print(english_op_val.shape, english_op_train.shape)
print(hindi_op_val.shape, hindi_op_train.shape)
print(mandarin_op_val.shape, mandarin_op_train.shape)
#-----------------------------------------------------------------------

x_train_data = np.concatenate((english_train, hindi_train, mandarin_train), axis=0)
x_val_data = np.concatenate((english_val, hindi_val, mandarin_val), axis=0)
print(x_train_data.shape)
print(x_val_data.shape)

y_train_data = np.concatenate((english_op_train, hindi_op_train, mandarin_op_train), axis=0)
y_val_data = np.concatenate((english_op_val, hindi_op_val, mandarin_op_val), axis=0)
print(y_train_data.shape)
print(y_val_data.shape)

y_weights = y_train_data[:, 0, 0]
class_weights = compute_class_weight('balanced', np.unique(y_weights), y_weights)
d_class_weights = dict(enumerate(class_weights))
print(d_class_weights)

weight_matrix = create_weights_matrix(y_train_data, d_class_weights)
print(weight_matrix.shape)


#-----------------------------------------------------------------------
#Shuffle data
np.random.seed(5)
indices = np.random.permutation(x_train_data.shape[0])
x_train_data = x_train_data[indices, :, :]
y_train_data = y_train_data[indices, :, :]

indices = np.random.permutation(x_val_data.shape[0])
x_val_data = x_val_data[indices, :, :]
y_val_data = y_val_data[indices, :, :]

#-----------------------------------------------------------------------

######  Define/Build/Train Training Model

training_in_shape = x_train_data.shape[1:]
training_in = Input(shape=training_in_shape)
foo = GRU(64, return_sequences=True, stateful=False)(training_in)
training_pred = Dense(3, activation='softmax')(foo)

training_model = Model(inputs=training_in, outputs=training_pred)
training_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode='temporal')
training_model.summary()

training_model.fit(x=x_train_data, y=y_train_data, validation_data=(x_val_data, y_val_data), batch_size=16, epochs=12, sample_weight=weight_matrix)

#------------------------------------------------------------------------

streaming_in = Input(batch_shape=(1, None, feature_dim))  ## stateful ==> needs batch_shape specified
foo = GRU(64, return_sequences=False, stateful=True )(streaming_in)
streaming_pred = Dense(3, activation='softmax')(foo)
streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

streaming_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode='temporal')
streaming_model.summary()

##### copy the weights from trained model to streaming-inference model
training_model.save_weights('weights.hd5', overwrite=True)
streaming_model.load_weights('weights.hd5')
streaming_model.save('streaming_model.hdf5')
plot_model(training_model, to_file='training_model.png', show_shapes=True, show_layer_names=True)
plot_model(streaming_model, to_file='streaming_model.png', show_shapes=True, show_layer_names=True)


