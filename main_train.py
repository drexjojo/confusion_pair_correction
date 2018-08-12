import sys
from code.get_cps import *
from code.make_confusion_dict import *
from code.make_dataset import *
from code.preprocessing import *
from code.train import *

language = input("Select language to train : ")


#Create confusion dict
get_cps(language)
print("[INFO] -> Finshed extracting confusion pairs !")
make_confusion_dict(language)
print("[INFO] -> Finshed creating confusion dict !")

#Create dataset 
make_dataset(language)
print("[INFO] -> Created Dataset!")

#Preprocessing
preprocessing(language)
print("[INFO] -> Preprocessing Done")

#training
train_model(language)
print("[INFO] -> Training Done")

