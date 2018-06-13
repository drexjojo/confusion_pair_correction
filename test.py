import numpy as np    
import pickle
import keras
import time
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate
from keras.models import model_from_json
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

with open('pre_trained_models/model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights("pre_trained_models/model_weight.hdf5")
# model = load_model('./pre_trained_models/model.h5')
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
EMBEDDING_DIM = 200
len_input_vocab = len(word_to_index.keys())
len_output_vocab = 4

def process(sentence,decoded_sentence) :
    ans = ""
    for i,pred in enumerate(decoded_sentence) :
        if pred != '3' and pred != '1':
            # print (i)
            ans = ans + sentence[i] + " "

    return ans


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

with open("text_files/eval_data.txt") as f:
    for line in f.readlines() :
        line = line.split("</>")[0]
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

    start_time = time.time()
    decoded_sentence = inference(input_seq)
    print("TIME : ",time.time()-start_time)
    answer = process(input_test_sentences[seq_index],decoded_sentence)
    print(input_test_sentences[seq_index])
    print(decoded_sentence)