This section contains the execution scripts for each part.

The training, validation, and test data are stored in the `data` folder within this directory (due to the large size of the training data, only a portion of it is included).

# Part2: Transformer Scaling Study

This section includes training nanoGPT models of various sizes on the dataset.

The `config` folder contains detailed configuration files for various sizes.

The `data` folder contains the dataset used for training.

`part2_train.sh` is a script that progressively trains nanoGPT models of different sizes and saves the results in the `scaling_logs` folder.

The `part2` folder contains the obtained results and the analysis conclusions based on these results.

# Part3: RNN Scaling Study and Comparison

This section involves training self-built RNNs of different sizes.

The `rnn_model.py` file contains the basic architecture of the RNN.

The `train_rnn.py` file contains instructions on how to train an RNN of a specific size.

`run_rnn_scaling.sh` is a script used to progressively train RNNs of different sizes and store the training results in the `part3` folder.

The `part3` folder contains all the training results for the RNN model and the analysis of these results, including a comparison between the RNN and nanoGPT models.

# Part4: Best Model Training and Sample Generation

In this section, I chose the results of training nanoGPT XL for three epochs on the dataset as the best weights for generating audio samples.

The `run_part4_epochs.sh` file contains instructions to train the nanoGPT XL model on the training set for ten epochs and save the results of each epoch into the `part4_results` folder.

The `eval_ckpt_val_test.py` file contains the code to re-evaluate the best weights on the validation and test datasets and generate the corresponding perplexity (PPL) scores.

The `part4_generate_samples.py` file is used to generate several ABC samples using the best weights, which are then converted into audio.

The `part4_results` folder contains all the generated ABC and audio samples, as well as the script used to convert the ABC files into audio samples.
