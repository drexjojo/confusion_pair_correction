import gensim
sentences = []
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

with open("./text_files/eng_sent.txt") as f:
	for line in f.readlines():
		line = tokenizer.tokenize(line)
		sentences.append(line)
print("[INFO] -> loaded sentences")

model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4,hs=1,negative=0)
model.save("pre_trained_models/language_model")


# print(model.score(["What is the name of your band?".split()]))

# print(model.score(["What is the name of you're band?".split()]))
# print(model.score(["What is your roll in this town?".split()]))
# print(model.score(["What is your role in this town?".split()]))