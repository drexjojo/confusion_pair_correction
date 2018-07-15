import pickle
from random import randint
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize

is_skew = True
def make_dataset(language) :

	with open(language+"/pickle_files/confusion_dict.pkl",'rb') as f :
		confusion_dict = pickle.load(f)
	data = []
	targets = []

	print("[INFO] -> Started creating dataset!")

	count = 0
	with open(language+"/text_files/targets.txt","w") as f3 :
		with open(language+"/text_files/data.txt","w") as f2 :
			with open(language+"/text_files/correct_sentences.txt") as f :
				for line in f.readlines():
					if count%10000 == 0 :
						print("Sentences done : ",count+1)
					count += 1
					tokenized_line = word_tokenize(line,language = language.lower())
					if len(tokenized_line) <= 20 :
						aug_sentences = [" ".join(tokenized_line)]
						aug_targets = [" ".join(['0' for i in range(len(tokenized_line))])]
						if is_skew == True :
							flag = False
						for i,word in enumerate(tokenized_line) :
							if word in confusion_dict.keys() :
								if is_skew == True :
									flag = True	
								temp_sentences = aug_sentences[:]
								for alt_word in confusion_dict[word] :
									if alt_word != word :
										for j,sent in enumerate(temp_sentences) :
											new_sent = word_tokenize(sent,language = language.lower())
											new_sent[i] = alt_word
											new_sent = " ".join(new_sent)
											aug_sentences.append(new_sent)
											new_target = aug_targets[j].split()
											new_target[i] = '1'
											new_target = " ".join(new_target)
											aug_targets.append(new_target)
						if is_skew == True : 
							if flag == False :
								if randint(0,10) == 2 :
									data = data + aug_sentences
									targets = targets + aug_targets
							if flag == True :
								data = data + aug_sentences
								targets = targets + aug_targets
						else :
							data = data + aug_sentences
							targets = targets + aug_targets

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

	




