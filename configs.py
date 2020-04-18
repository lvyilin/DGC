config_celeba = {}
config_celeba['dataset'] = 'celeba'
config_celeba['verbose'] = True
config_celeba['save_every_epoch'] = 10
config_celeba['print_every'] = 1000

config_celeba['adam_beta1'] = 0.5
config_celeba['lr'] = 5e-5
config_celeba['lr_schedule'] = 'manual_smooth'  # manual, plateau, or a number
config_celeba['batch_size'] = 25
config_celeba['epoch_num'] = 30
config_celeba['init_std'] = 0.0099999
config_celeba['init_bias'] = 0.0
config_celeba['batch_norm'] = True
config_celeba['batch_norm_eps'] = 1e-05
config_celeba['batch_norm_decay'] = 0.9
config_celeba['conv_filters_dim'] = 5

config_celeba['e_pretrain'] = True
config_celeba['e_pretrain_sample_size'] = 256
config_celeba['e_num_filters'] = 1024
config_celeba['e_num_layers'] = 5
config_celeba['g_num_filters'] = 1024
config_celeba['g_num_layers'] = 4
config_celeba['d_num_layers'] = 4
config_celeba['d_num_filters'] = 1024

config_celeba['zdim'] = 64
config_celeba['cost'] = 'l2sq'  # l2, l2sq, l1
config_celeba['lambda'] = 1.

config_celeba['n_classes'] = 5
config_celeba['hdim'] = 2 * 2 * 1024

config_celeba['augment_z'] = True
config_celeba['augment_x'] = False
config_celeba['LVO'] = False
config_celeba['eval_strategy'] = 2
config_celeba['aug_rate'] = 0.6
config_celeba['mlp_classifier'] = True
config_celeba['sampling_size'] = 10

config_mnist = {}
config_mnist['dataset'] = 'mnist'
config_mnist['verbose'] = True
config_mnist['save_every_epoch'] = 10
config_mnist['print_every'] = 100

config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 1e-5
config_mnist['lr_schedule'] = 'manual'  # manual, plateau, or a number
config_mnist['batch_size'] = 100
config_mnist['epoch_num'] = 50
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9
config_mnist['conv_filters_dim'] = 4

config_mnist['e_pretrain'] = True
config_mnist['e_pretrain_sample_size'] = 1000
config_mnist['e_num_filters'] = 1024
config_mnist['e_num_layers'] = 4
config_mnist['g_num_filters'] = 1024
config_mnist['g_num_layers'] = 3

config_mnist['d_num_filters'] = 512
config_mnist['d_num_layers'] = 4

config_mnist['zdim'] = 64
config_mnist['cost'] = 'l2sq'  # l2, l2sq, l1
config_mnist['lambda'] = 1.

config_mnist['n_classes'] = 10
config_mnist['hdim'] = 2 * 2 * 1024

config_mnist['mlp_classifier'] = False
config_mnist['eval_strategy'] = 1
config_mnist['sampling_size'] = 10
config_mnist['augment_z'] = True
config_mnist['augment_x'] = False
config_mnist['aug_rate'] = 0.6
config_mnist['LVO'] = True
