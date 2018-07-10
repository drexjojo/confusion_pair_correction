# confusion_pair_correction

Instructions for training
------

> 1) Choose the language to train on. There should be a directory with the same name as the language to train in the main directory. 
> 2) If the dataset is skewed (more correct samples than incorrect samples) Change the `is_skew` variable to `True` in code/make_dataset.py
> 3) If you are using pretrained word embedding, specify the correct path in code/train.py
> 4) Specify the number of GPUs and CPUs to be used in code/train.py
> 5) run `python3 main_train.py`
