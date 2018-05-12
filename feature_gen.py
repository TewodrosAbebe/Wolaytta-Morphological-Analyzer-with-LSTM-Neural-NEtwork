import numpy as np
from process_data import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os

# Read the data from text file
data = get_raw_data('text.txt')
result = setup_features_and_targets(data)
# Create input and out for seq2seq LSTM encoder and decoder
X_encoder, X_decoder, Y = setup_data_for_seq2seq(
    result[0], result[1], result[2], result[3], result[4])
encoder_size = X_encoder.shape[2]
decoder_size = X_decoder.shape[2]


def define_models(n_input, n_output, n_units):
    """ Create a model containing Encoder and Decoder

    Arguments:
    n_inputs -- the Sequence length of the input

    n_output -- the Sequence length of the output and the input to the decoder

    n_units -- the LSTM cells

    Return:
    model -- Seq2Seq model

    encoder_model & decoder_model -- models for prediction
    """
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


train, infenc, infdec = define_models(encoder_size, decoder_size, 256)
train.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

train.fit([X_encoder, X_decoder], Y, epochs=50)
