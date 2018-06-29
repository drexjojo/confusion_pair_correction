length = []
with open("text_files/BNC_corpus.txt") as f:
	for line in f.readlines() :
		length.append(len(line.split()))


summ = 0
for i in length :
	summ += i 

print(summ/len(length))