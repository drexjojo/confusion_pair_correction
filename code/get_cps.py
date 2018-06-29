def get_cps(language) :
	with open(language+"/text_files/selected_cps.txt",'w') as g:
		with open(language+"/text_files/LT_cps.txt") as f:
			for line in f.readlines() :
				line = line.split(";")
				first = line[0].split("|")[0].strip().lower()
				last = line[1].split("|")[0].strip().lower()
				g.write(first+" "+last+" \n")

	