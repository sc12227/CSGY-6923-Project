# Part 1 Data Collection and Preprocessing

This part mainly includes downloading data, cleaning data, and splitting the data into training and testing sets.

Data download address: http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz

After downloading the dataset, place the entire dataset in the `../data/midi_raw` folder.

Execution order:

`midi_to_abc_mp.py`: Convert the MIDI dataset to ABC dataset.

`delete_too_long _abc.py`: Remove datasets with unreasonable lengths.

`clean_abc_raw_index_by_token.py`: Remove datasets with unreasonable toekns.

`build_1b_index.py`:Select a dataset with at least 1 billion tokens.

`build_vocab.py`:Build a vocabulary list

`split_abc_by_token_count.py`：Splitting the data into training and testing sets.
