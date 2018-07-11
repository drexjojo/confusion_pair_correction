# confusion_pair_correction

Instructions For Training
------

> 1) Choose the language to train on. There should be a directory with the same name as the language to train in the main directory. 
> 2) If the dataset is skewed (more correct samples than incorrect samples) Change the `is_skew` variable to `True` in code/make_dataset.py
> 3) If you are using pretrained word embedding, specify the correct path in code/train.py
> 4) Specify the number of GPUs and CPUs to be used in code/train.py
> 5) Put the LT confusion pairs for a language in _language_/text_files/LT_cps.txt. This file should be of the format described [here](https://github.com/languagetool-org/languagetool/blob/master/languagetool-language-modules/en/src/main/resources/org/languagetool/resource/en/confusion_sets.txt)
> 6) The file _language_/text_files/correct_sentences.txt contains all the correct sentences that the model uses for training.
> 7) run `python3 main_train.py`

File Structure
------
> 1) _language_/text_files/selected_cps.txt contains all the confusion pairs supported.
> 2) _language_/text_files/data.txt contains all the correct+incorrect sentences and _language_/text_files/targets.txt contains the corresponding targets used for training.
> 3) _language_/pickle_files contains all the pickle files created.
> 4) _language_/pre_trained_models/ contains the trained model files and the weight files.
