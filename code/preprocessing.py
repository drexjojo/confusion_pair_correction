from sklearn.utils import shuffle
from keras.preprocessing import sequence
import nltk
import itertools
import pickle
import numpy as np

VOCAB_SIZE = 40000
MAX_SEQUENCE_LENGTH = 20
MAX_DECODER_SEQUENCE_LENGTH = 20
len_output_vocab = 4

def preprocessing(language) :
	data = []
	targets = []

	print("[INFO] -> Reading files")
	with open(language+"/text_files/data.txt") as f :
		for line in f.readlines() :
			data.append(line.split())

	with open(language+"/text_files/targets.txt") as f :
		for line in f.readlines() :
			mod_line = "2 "+line+" 3"
			targets.append(mod_line.split())

	print("[INFO] -> Done!")

	np_data = np.array(data)
	np_targets = np.array(targets)
	np_data,np_targets = shuffle(np_data,np_targets)
			
	temp_input = [word for k in np_data for word in k ]
	input_word_freq = nltk.FreqDist(itertools.chain(temp_input))
	input_vocab = input_word_freq.most_common(VOCAB_SIZE-1)

	temp_output = [word for k in np_targets for word in k ]
	output_word_freq = nltk.FreqDist(itertools.chain(temp_output))
	output_vocab = output_word_freq.most_common(VOCAB_SIZE-1)

	input_index_to_word = [x[0] for x in input_vocab]
	input_index_to_word.append("UNK")
	word_to_index = dict([(w,i) for i,w in enumerate(input_index_to_word)])

	for i, sent in enumerate(np_data):
		np_data[i] = [w if w in word_to_index.keys() else "UNK" for w in sent]

	X = np.asarray([[word_to_index[w] for w in sent] for sent in np_data])
	Y = np.asarray([[w for w in sent] for sent in np_targets])

	padded_input = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	padded_output = sequence.pad_sequences(Y, maxlen=MAX_SEQUENCE_LENGTH, padding='post',value = 3)

	decoder_target_data = np.zeros((len(padded_output), MAX_DECODER_SEQUENCE_LENGTH, len_output_vocab),dtype='float32')
	decoder_input_data = np.zeros((len(padded_output), MAX_DECODER_SEQUENCE_LENGTH, len_output_vocab),dtype='float32')

	for i, target_text in enumerate(padded_output):
	    for t, char in enumerate(target_text):
	        decoder_input_data[i, t, int(char)] = 1.
	        if t > 0:
	            decoder_target_data[i, t - 1, int(char)] = 1.


	f = open(language+"/pickle_files/word_to_index.pkl" , 'wb')
	pickle.dump(word_to_index,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	f = open(language+"/pickle_files/padded_input.pkl" , 'wb')
	pickle.dump(padded_input,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	f = open(language+"/pickle_files/padded_output.pkl" , 'wb')
	pickle.dump(padded_output,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	f = open(language+"/pickle_files/decoder_input_data.pkl" , 'wb')
	pickle.dump(decoder_input_data,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	f = open(language+"/pickle_files/decoder_target_data.pkl" , 'wb')
	pickle.dump(decoder_target_data,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

	