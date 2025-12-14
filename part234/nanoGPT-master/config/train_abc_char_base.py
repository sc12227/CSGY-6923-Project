out_dir = 'out-abc-char-base'   

eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'abc-char-scaling'
wandb_run_name = 'debug-run'

dataset = 'abc_char'       

gradient_accumulation_steps = 4
batch_size = 32
block_size = 512          

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

bias = False               
vocab_size = None          

learning_rate = 3e-4
max_iters = 15000          
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Cosine LR decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 3e-5

backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'   
compile = True
