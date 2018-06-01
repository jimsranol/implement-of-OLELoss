import os

use_OLE = True
lambda_ = 0.1

data_dir = 'hymenoptera_data'

use_pretrained = True
is_resume = False
checkpoint = os.path.join('weight', 'OLE+Cross')

class_num = 2
max_epoch = 20
batch_size = 64

initial_lr = 0.001
decay_step = 10
decay_scalar = 0.1
