python ./preprocess_ohmnet.py
gzip ../data/kg/*
python ./buid_dataset.py
gzip ../data/dataset/*