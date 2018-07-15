import sys
from code.evaluation import *

language = input("Select language to evaluate : ")
#language = "French"


#Create Evaluation Data
make_evaluation_data(language+"/text_files/evaluation_correct_sentences.txt",language)
exit(0)

#Evaluate the model
evaluate(language)
print("[INFO] -> Evaluation Done !")

