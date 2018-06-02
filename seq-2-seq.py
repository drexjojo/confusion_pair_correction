import numpy as np    
import pickle
import keras
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate

config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 40} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
# from keras.backend import manual_variable_initialization 
# manual_variable_initialization(True)

with open("./pickle_files/confusion_dict.pkl",'rb') as f:
    confusion_dict = pickle.load(f)

with open("./pickle_files/padded_input.pkl",'rb') as f:
    padded_input = pickle.load(f)

with open("./pickle_files/padded_output.pkl",'rb') as f:
    padded_output = pickle.load(f)

with open("./pickle_files/word_to_index.pkl",'rb') as f:
    word_to_index = pickle.load(f)

with open("./pickle_files/decoder_input_data.pkl",'rb') as f:
    decoder_input_data = pickle.load(f)

with open("./pickle_files/decoder_target_data.pkl",'rb') as f:
    decoder_target_data = pickle.load(f)

def find_other(word) :
    for lis in confusion_pairs :
        if word in lis :
            for word2 in lis :
                if word2 != word :
                    return word2


confusion_words = confusion_dict.keys()
MAX_ENCODER_SEQUENCE_LENGTH = 20
MAX_DECODER_SEQUENCE_LENGTH = 20
LATENT_DIM = 300
BATCH_SIZE = 100
EPOCHS = 1
EMBEDDING_DIM = 200
len_input_vocab = len(word_to_index.keys())
len_output_vocab = 4

print("[INFO] -> Data Loaded")



#------------- UNCOMMENT TO TRAIN THE MODEL --------------------------------------------

# print("[INFO] -> Starting Training")

# UNCOMMETNT TO USE ONE HOT EMBEDDINGS
# encoder_inputs = Input(shape=(None,))
# x = Embedding(len_input_vocab, LATENT_DIM)(encoder_inputs)
# x, state_h, state_c = LSTM(LATENT_DIM,return_state=True)(x)
# encoder_states = [state_h, state_c]


# UNCOMMENT TO USE GLOVE EMBEDDINGS
# embeddings_index = {}
# f = open('glove.6B.200d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

# print('[INFO] - > Found %s word vectors.' % len(embeddings_index))

# embedding_matrix = np.zeros((len_input_vocab, EMBEDDING_DIM))
# for word, i in word_to_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# embedding_layer = Embedding(len_input_vocab,EMBEDDING_DIM,weights=[embedding_matrix],
#                             input_length=MAX_ENCODER_SEQUENCE_LENGTH,
#                             trainable=False)

# encoder_inputs = Input(shape=(None,))
# x = embedding_layer(encoder_inputs)
# x, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(LATENT_DIM,return_state=True))(x)
# state_h = Concatenate()([forward_h, backward_h])
# state_c = Concatenate()([forward_c, backward_c])
# encoder_states = [state_h, state_c]


# decoder_inputs = Input(shape=(None, len_output_vocab))
# decoder_lstm = LSTM(LATENT_DIM * 2, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# decoder_dense = Dense(len_output_vocab, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)


# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer='Adam', loss='categorical_crossentropy')

# print(model.summary())

# model.fit([padded_input, decoder_input_data], decoder_target_data,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCHS,
#           validation_split=0.05)

# with open('./pre_trained_models/model.json', 'w') as f:
#     f.write(model.to_json())
# model.save('./pre_trained_models/model.h5')
# model.save_weights('./pre_trained_models/model_weights.h5')

# encoder_model = Model(encoder_inputs, encoder_states)
# decoder_state_input_h = Input(shape=(LATENT_DIM*2,))
# decoder_state_input_c = Input(shape=(LATENT_DIM*2,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)

# encoder_model.compile(optimizer='Adam', loss='categorical_crossentropy')
# decoder_model.compile(optimizer='Adam', loss='categorical_crossentropy')

# with open('./pre_trained_models/encoder_model.json', 'w') as f:
#     f.write(encoder_model.to_json())
# encoder_model.save('./pre_trained_models/encoder_model.h5')
# encoder_model.save_weights('./pre_trained_models/encoder_model_weights.h5')

# with open('./pre_trained_models/decoder_model.json', 'w') as f:
#     f.write(decoder_model.to_json())
# decoder_model.save('./pre_trained_models/decoder_model.h5')
# decoder_model.save_weights('./pre_trained_models/decoder_model_weights.h5')
# print("[INFO] -> Training Done")


#-----------UNCOMMENT TO USE PRE-TRAINED MODELS ---------------------------------
model = load_model('./pre_trained_models/model.h5')
# model.load_weights('./pre_trained_models/model_weights.h5')
encoder_model = load_model("./pre_trained_models/encoder_model.h5")
# encoder_model.load_weights('./pre_trained_models/encoder_model_weights.h5')
decoder_model = load_model("./pre_trained_models/decoder_model.h5")
# decoder_model.load_weights('./pre_trained_models/decoder_model_weights.h5')
print("[INFO] -> Trained models loaded")
#---------------------------------------------------------------------------------

def inference(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, len_output_vocab))
    target_seq[0, 0, 2] = 1.
    flag = False
    decoded_sentence = ''
    while not flag:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = sampled_token_index
        decoded_sentence += str(sampled_char)

        if (sampled_char == '3' or len(decoded_sentence) > MAX_DECODER_SEQUENCE_LENGTH):
            flag = True

        target_seq = np.zeros((1, 1, len_output_vocab))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence



#-------------------- TESTING-----------------------------------------------------
input_test_sentences = []

print("[INFO] -> Starting Testing")

with open("test_set.txt") as f:
    for line in f.readlines() :
        line = line.replace(".","").replace("!","").replace("?","").replace(",","").lower()
        input_test_sentences.append(line.split())

input_test = input_test_sentences[:]

for i, sent in enumerate(input_test_sentences):
    input_test[i] = [w if w in word_to_index.keys() else "UNK" for w in sent]

X = np.asarray([[word_to_index[w] for w in sent] for sent in input_test])
padded_input_test = sequence.pad_sequences(X, maxlen=MAX_ENCODER_SEQUENCE_LENGTH)


generated_ans = []
for seq_index in range(len(input_test)):
    input_seq = padded_input_test[seq_index: seq_index + 1]
    decoded_sentence = inference(input_seq)
    answer = []
    for i in decoded_sentence:
        if i == '3' :
            break 
        else :
            answer.append(i)

    if '1' not in answer:
        #case where there is no error predicted by the model
        print('-')
        print('Input sentence  :', " ".join(input_test_sentences[seq_index]))
        print('Output sentence :', " ".join(input_test_sentences[seq_index]))
        print("\n")

    else :
        predicted_output = []
        if len(answer) == len(input_test[seq_index]) :
            for i,tar in enumerate(answer) :
                if tar == '0' :
                    predicted_output.append(input_test_sentences[seq_index][i])
                else :
                    if input_test[seq_index][i] in confusion_words :
                        predicted_output.append(find_other(input_test_sentences[seq_index][i]))
                    else :
                        predicted_output.append(input_test_sentences[seq_index][i])

            generated_ans.append(" ".join(predicted_output))
            print('-')
            print('Input sentence  :', " ".join(input_test_sentences[seq_index]))
            print('Output sentence :', " ".join(predicted_output))
            # print('supposed to be : ', " ".join(target_test[seq_index][1:-1]))
            print("\n")
        else :
            #failed output : print originial sentence
            print('-')
            # print("Model failed :( ")
            print('Input sentence  :', " ".join(input_test_sentences[seq_index]))
            print('Output sentence :', " ".join(input_test_sentences[seq_index]))
            generated_ans.append(" ".join(input_test_sentences[seq_index]))
            print("\n")

# with open("generated_ans.txt",'w') as f:
# 	for line in generated_ans :
# 		f.write(line + "\n")

# for line in generated_ans :
#     print(line)
