import pickle

def make_confusion_dict(language) :
	confusion_dict = {}
	with open(language+"/text_files/selected_cps.txt") as f:
		for line in f.readlines() :
			line = line.split()
			first = line[0].lower()
			last = line[1].lower()

			if first in confusion_dict.keys() :
				if last not in confusion_dict[first] :
					confusion_dict[first].append(last)
			else :
				confusion_dict[first] = [last]

			if last in confusion_dict.keys() :
				if first not in confusion_dict[last] :
					confusion_dict[last].append(first)
			else :
				confusion_dict[last] = [first]


	f = open(language+"/pickle_files/confusion_dict.pkl" , 'wb')
	pickle.dump(confusion_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

