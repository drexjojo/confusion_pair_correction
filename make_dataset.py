import pickle
from random import randint
from nltk.tokenize import TweetTokenizer

with open("./pickle_files/confusion_dict.pkl",'rb') as f :
	confusion_dict = pickle.load(f)

tokenizer = TweetTokenizer()

data = []
targets = []

print("[INFO] -> Started!")

count = 0
with open("text_files/targets.txt","w") as f3 :
	with open("text_files/data.txt","w") as f2 :
		with open("text_files/eng_sent.txt") as f :
			for line in f.readlines():
				print(count+1)
				count += 1
				tokenized_line = tokenizer.tokenize(line)
				if len(tokenized_line) <= 20 :
					aug_sentences = [" ".join(tokenized_line)]
					aug_targets = [" ".join(['0' for i in range(len(tokenized_line))])]
					for i,word in enumerate(tokenized_line) :
						if word in confusion_dict.keys() :
							temp_sentences = aug_sentences[:]
							for alt_word in confusion_dict[word] :
								if alt_word != word :
									for j,sent in enumerate(temp_sentences) :
										new_sent = tokenizer.tokenize(sent)
										new_sent[i] = alt_word
										new_sent = " ".join(new_sent)
										aug_sentences.append(new_sent)
										new_target = aug_targets[j].split()
										new_target[i] = '1'
										new_target = " ".join(new_target)
										aug_targets.append(new_target)

					data = data + aug_sentences
					targets = targets + aug_targets

				# else :
				# 	data.append(" ".join(tokenized_line))
				# 	targets.append(" ".join(['0' for i in range(len(tokenized_line))]))
				# 	new_sent = tokenized_line[:]
				# 	new_target = ['0' for i in range(len(tokenized_line))]
				# 	for i,word in enumerate(tokenized_line) :
				# 		if word in confusion_dict.keys() :
				# 			randomint = randint(0,len(confusion_dict[word]))
				# 			if randomint != 0 :
				# 				new_target[i] = '1'
				# 				new_sent[i] = confusion_dict[word][randomint-1]
				# 	new_sent = " ".join(new_sent)
				# 	new_target = " ".join(new_target)
				# 	data.append(new_sent)	
				# 	targets.append(new_target)

				if len(data) > 100000 :
					for data_point in data :
						f2.write(data_point + "\n")
					data = []

				if len(targets) > 100000 :
					for data_point in targets :
						f3.write(data_point + "\n")
					targets = []
		if len(data) > 0 :
			for data_point in data :
				f2.write(data_point + "\n")
			data = []

		if len(targets) > 0 :
			for data_point in targets :
				f3.write(data_point + "\n")
			targets = []

print("[INFO] -> Done!")




