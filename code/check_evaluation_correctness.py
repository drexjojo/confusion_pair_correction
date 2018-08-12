from nltk.tokenize import word_tokenize

with open("French/text_files/eval_targets.txt") as f:
	tar = f.readlines() 
with open("French/text_files/eval_data.txt") as f:
	for i,line in enumerate(f.readlines()) :
		lin = line.split("</>")[0].strip().split()
		if len(lin) != len(tar[i].split()) :
			print(i)
			print(line)
			print(lin)
			print(tar[i].split())
			# print(le)
			print(len(line))
			print(len(lin))
			# print(len(le))
			print(len(tar[i].split()))
			break
