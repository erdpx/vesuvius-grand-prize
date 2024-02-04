from segments_config import *

config = {
  'train': False,
  'infer': True,

  'segmentation': False,

  '3dmodel': True,
  'model': 'mednext',

  'model_info': 'zdim32_mednext',

  'load_checkpoint': False,
  'checkpoint_path': '',
  'save_checkpoint': True,

  'segments_path': '',
  'labels_path': '',
  'noink_labels_path': '',

  'num_epochs': 150,
  'batch_size': 16,
  'infer_batch_size': 128,
  'learning_rate': 1e-3,

  'z_dim': 32,
  'z_start': False,
  'tile_size': 64,
  'stride': 32,
  'infer_stride': 32,
}

aug = {
  'size': config['tile_size'],
  'in_channels': config['z_dim'],
}
