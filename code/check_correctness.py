from nltk.tokenize import word_tokenize

with open("English/text_files/targets.txt") as f:
	tar = f.readlines() 
with open("English/text_files/data.txt") as f:
	for i,line in enumerate(f.readlines()) :
		lin = line.split()
		le = word_tokenize(line,language = "french")
		if len(lin) != len(tar[i].split()) :
			print(i)
			print(line)
			print(lin)
			print(tar[i].split())
			print(le)
			print(len(line))
			print(len(lin))
			print(len(le))
			print(len(tar[i].split()))
			break
