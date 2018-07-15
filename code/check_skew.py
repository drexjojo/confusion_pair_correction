count = 0
with open("../French/text_files/targets.txt") as f :
	for line in f.readlines() :
		if "1" not in line :
			count += 1
print(count)