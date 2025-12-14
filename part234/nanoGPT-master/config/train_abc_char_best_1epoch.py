from config.train_abc_char_base import *

out_dir = 'DUMMY_OUT_DIR'

n_layer = 12
n_head  = 10
n_embd  = 640
dropout = 0.1

max_iters = 15000    
lr_decay_iters = max_iters

learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1

eval_interval = 15000     
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
