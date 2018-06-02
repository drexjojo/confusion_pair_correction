import numpy as np    
import pickle
import keras
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

model = load_model('./pre_trained_models/model.h5')
# model.load_weights('./pre_trained_models/model_weights.h5')
encoder_model = load_model("./pre_trained_models/encoder_model.h5")
# encoder_model.load_weights('./pre_trained_models/encoder_model_weights.h5')
decoder_model = load_model("./pre_trained_models/decoder_model.h5")
# decoder_model.load_weights('./pre_trained_models/decoder_model_weights.h5')
print("[INFO] -> Trained models loaded")

with open("./pickle_files/word_to_index.pkl",'rb') as f:
    word_to_index = pickle.load(f)

with open("./pickle_files/confusion_dict.pkl",'rb') as f:
    confusion_dict = pickle.load(f)


confusion_words = confusion_dict.keys()
MAX_ENCODER_SEQUENCE_LENGTH = 20
MAX_DECODER_SEQUENCE_LENGTH = 20
LATENT_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 100
EMBEDDING_DIM = 200
len_input_vocab = len(word_to_index.keys())
len_output_vocab = 4


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

with open("text_files/test_set.txt") as f:
    for line in f.readlines() :
        tokenized_line = tokenizer.tokenize(line)
        input_test_sentences.append(tokenized_line)

input_test = input_test_sentences[:]

for i, sent in enumerate(input_test_sentences):
    input_test[i] = [w if w in word_to_index.keys() else "UNK" for w in sent]

X = np.asarray([[word_to_index[w] for w in sent] for sent in input_test])
padded_input_test = sequence.pad_sequences(X, maxlen=MAX_ENCODER_SEQUENCE_LENGTH)


generated_ans = []
for seq_index in range(len(input_test)):
    input_seq = padded_input_test[seq_index: seq_index + 1]
    decoded_sentence = inference(input_seq)
    print(input_test_sentences[seq_index])
    print(decoded_sentence)