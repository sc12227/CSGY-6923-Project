# CSGY-6923-Project

# Shuhan Cai

# sc12227@nyu.edu

# `project.pdf` 

This project is based on the Lakh MIDI Dataset.  It uses nanoGPT and a self-built RNN to train on this dataset, generating several results. This project analyzes the different performance of these two architectures on the Lakh MIDI Dataset and selects nanoGPT XL as the best option to produce the final results.

The dataset can be downloaded from: http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz

The final generated audio file is located at `CSGY-6923-Project/part234/nanoGPT-master/part4_results/midi_out_cond`

On this page, Part 1 contains multiple scripts for cleaning and removing unnecessary data from the dataset and splitting it into training sets. Parts 234 contain the results of multiple scripts that trained different sizes of nanoGPT and my own custom-built RNN on the training set created in Part 1, along with several analysis graphs. This part also selects nanoGPT XL as the final model and generates multiple audio samples.

For detailed instructions, please refer to the Readme.md file in each directory.

## Setup
pip install -r requirements.txt
