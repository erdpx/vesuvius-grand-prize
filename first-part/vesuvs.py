# %% [markdown]
#   ## SETUP

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from binary_dice_loss import BinaryDiceLoss
import numpy as np
import PIL.Image as Image
from tifffile import tifffile
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import sys
import time
import random


if len(sys.argv) == 1:
  import config
elif sys.argv[0].split('/')[-1] == 'ipykernel_launcher.py':
  import config as config
elif len(sys.argv) == 2:
  config = __import__(sys.argv[1])


# %%
Image.MAX_IMAGE_PIXELS = None


# %%
tile_size = config.config['tile_size'] if config.config['tile_size'] else 64
stride = config.config['stride'] if config.config['stride'] else tile_size // 4
infer_stride = config.config['infer_stride'] if config.config['infer_stride'] else 32

Z_DIM = config.config['z_dim'] if config.config['z_dim'] else 27
Z_START = config.config['z_start'] if config.config['z_start'] else 64 // 2 - Z_DIM // 2

NUM_EPOCHS = config.config['num_epochs'] if config.config['num_epochs'] else 5
BATCH_SIZE = config.config['batch_size'] if config.config['batch_size'] else 32
INFER_BATCH_SIZE = config.config['infer_batch_size'] if config.config['infer_batch_size'] else 32
LEARNING_RATE = config.config['learning_rate'] if config.config['learning_rate'] else 0.001

SEGMENTS_PATH = config.config['segments_path'] if config.config['segments_path'] else f'PATH_TO_SEGMENTS'
LABELS_PATH = config.config['labels_path'] if 'labels_path' in config.config else os.path.join(SEGMENTS_PATH, 'labels')
NOINK_LABELS_PATH = config.config['noink_labels_path'] if 'noink_labels_path' in config.config else os.path.join(SEGMENTS_PATH, 'noinklabels')

SEGMENTATION = config.config['segmentation'] if 'segmentation' in config.config else False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# ## AUGMENTATIONS

# %%
DYNAMIC_AUG_P = True
t_start = 0
if DYNAMIC_AUG_P:
  from scipy.integrate import solve_ivp
  def SIRSModel(t, state, beta, gamma, zeta):
    S, I, R = state
    N = S + I + R
    dSdt = -beta*S*I/N + zeta*R
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I - zeta*R
    return [dSdt, dIdt, dRdt]

  initial_state = [127, 1, 0]
  N = sum(initial_state)
  t_span = (t_start, t_start+NUM_EPOCHS+200)
  t_eval = np.linspace(t_span[0], t_span[1], NUM_EPOCHS+200)

  solP = solve_ivp(SIRSModel, t_span, initial_state, t_eval=t_eval, args=(0.55, 0.1, 0.2))
  # solP = solve_ivp(SIRSModel, t_span, initial_state, t_eval=t_eval, args=(0.35, 0.05, 0))
  solP.y /= N
  training_aug_p = np.concatenate((t_start*[0], solP.y[1]))
else:
  training_aug_p = np.concatenate((t_start*[0], NUM_EPOCHS*[0.5]))

# %%
def get_training_aug_list(prob):
  training_aug_list = [
    A.HorizontalFlip(p=prob),
    A.VerticalFlip(p=prob),
    # A.RandomRotate90(),
    A.RandomBrightnessContrast(p=prob, brightness_limit=.2, contrast_limit=prob),
    # A.ChannelDropout(channel_drop_range=(1,4), p=prob - 0.2 if prob - 0.2 > 0 else prob),
    A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.2, scale_limit=0.2, p=prob+0.2),
    A.OneOf([
      A.GaussNoise(var_limit=[10,50]),
      A.GaussianBlur(),
      A.MotionBlur()
    ], p=prob - 0.1 if prob - 0.1 > 0 else prob),
    # A.ElasticTransform(p=prob),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=prob),
    # A.CoarseDropout(max_holes=2, max_width=int(0.05*config.aug['size']), max_height=int(0.05*config.aug['size']), mask_fill_value=0, p=prob),
    A.Normalize(
      mean=[0]*config.aug['in_channels'],
      std=[1]*config.aug['in_channels']
    ),
    ToTensorV2(transpose_mask=True)
  ]

  return training_aug_list


validation_aug_list = [
  A.Normalize(
    mean=[0]*config.aug['in_channels'],
    std=[1]*config.aug['in_channels']
  ),
  ToTensorV2(transpose_mask=True)
]

# %%
def get_transforms(transformType, prob=0):
  if transformType == 'training':
    if prob:
      return A.Compose(get_training_aug_list(prob))
    else:
      return A.Compose(get_training_aug_list(0.5))
  elif transformType == 'validation':
    return A.Compose(validation_aug_list)

# %% [markdown]
#   ## DATASET

# %%
class SubvolumeDataset(data.Dataset):
  def __init__(self, image_stack, label, pixels, segment_id, transforms=False):
    self.image_stack = image_stack
    self.label = label
    self.pixels = pixels
    self.segment_id = segment_id
    self.transforms = transforms

  def __len__(self):
    return len(self.pixels)
  
  def __getitem__(self, index):
    x1, y1, x2, y2 = self.pixels[index]
    subvolume = np.stack([torch.from_numpy(image[y1:y2, x1:x2]) for image in self.image_stack], axis=2)

    if self.transforms:
      augmented_data = self.transforms(image=subvolume)
      subvolume = augmented_data['image'].unsqueeze(0)
    else:
      subvolume = torch.from_numpy(subvolume)
      subvolume = subvolume.permute(2, 0, 1).unsqueeze(0)

    return subvolume, self.label

# %% [markdown]
#   ## MODEL

# %% [markdown]
#   ### 3D MODELS (1 CLASS PREDICTION)

# %%
if config.config['3dmodel'] and config.config['model'] == 'i3d':
  from i3dall import InceptionI3d
  model = InceptionI3d(in_channels=1,  num_classes=1)
  model.avg_pool.kernel_size = (1,2,2)
  # model.Conv3d_1a_7x7.conv3d.kernel_size = (1,2,2)
  model.Conv3d_1a_7x7.conv3d.kernel_size = (3,3,3)
  # model.Conv3d_1a_7x7.conv3d.stride = (2,2,2)
  model.Conv3d_1a_7x7.conv3d.stride = (1,1,1)
  model.Conv3d_1a_7x7.conv3d.padding = (2,2,2)
  model.forward_features = False

# %%
if config.config['3dmodel'] and config.config['model'] == 'mednext':
  from nnunet_mednext import MedNeXt
  model = MedNeXt(in_channels=1, n_channels=32, n_classes=1, kernel_size=3, deep_supervision=False, do_res=True, do_res_up_down=True, dim='3d', exp_r=[2,3,4,4,4,4,4,3,2], block_counts = [2,2,2,2,2,2,2,2,2])


# %% [markdown]
#   ### GENERAL MODEL MANIPULATION

# %%
model.to(DEVICE)


# %%
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)


# %%
if config.config['3dmodel'] and config.config['load_checkpoint'] == False:
  import weight_rewiring
  for m in model.modules():
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      weight_rewiring.PA_rewiring_np(m.weight)


# %% [markdown]
#   ## UTILS

# %%
def load_mask_label(segment_id, load_label=True):
  SEGMENT_PATH = os.path.join(SEGMENTS_PATH, segment_id)
  mask = np.array(Image.open(os.path.join(SEGMENT_PATH,f"{segment_id}_mask.png")).convert('1'))

  if load_label:
    label = np.array(Image.open(os.path.join(LABELS_PATH,f"{segment_id}.png")).convert('1'))
    
    label = label/255
    return torch.from_numpy(mask).float(), label
  else:
    return torch.from_numpy(mask).float(), mask

def load_noinklabel(segment_id=True):
  label = np.array(Image.open(os.path.join(NOINK_LABELS_PATH,f"{segment_id}.png")).convert('1'))
  return label

def load_images(segment_id):
  SEGMENT_PATH = os.path.join(SEGMENTS_PATH, segment_id)
  images = [tifffile.imread(filename).astype(np.float32)/65535.0 for filename in tqdm(sorted(glob.glob(os.path.join(SEGMENT_PATH,"layers/*.tif")))[Z_START:Z_START+Z_DIM])]
  
  if segment_id not in config.inverted:
    return images
  else:
    return images[::-1]


# %% [markdown]
#   ## TRAINING

# %%
segments_id = config.training_segments_id

# %%
criterion = nn.BCEWithLogitsLoss()
dice_loss = BinaryDiceLoss()

# scaler = torch.cuda.amp.GradScaler()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)



# %%
if config.config['load_checkpoint']:
  checkpoint = torch.load(config.config['checkpoint_path'])
  # model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']
  running_loss = checkpoint['running_loss']
  epoch_loss = checkpoint['epoch_loss']
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  new_state_dict = checkpoint['model_state_dict']
  if torch.cuda.device_count() == 1:
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]
        if name == 'module.':
          new_state_dict[name] = v
          
  model.load_state_dict(new_state_dict)

  model.to(DEVICE)

for param_group in optimizer.param_groups:
  param_group['lr'] = LEARNING_RATE

# %%
model_name = config.config['model'] + '_' + config.config['model_info']

model_checkpoints_path = './checkpoints'
if not os.path.isdir(model_checkpoints_path):
  os.mkdir(model_checkpoints_path)

if config.config['3dmodel']:
  model_checkpoints_path = os.path.join(model_checkpoints_path, '3d')
  if not os.path.isdir(model_checkpoints_path):
    os.mkdir(model_checkpoints_path)

elif config.config['2dmodel']:
  model_checkpoints_path = os.path.join(model_checkpoints_path, '2d')
  if not os.path.isdir(model_checkpoints_path):
    os.mkdir(model_checkpoints_path)
  
  if config.config['2d_in_channels']:
    model_preds_path = os.path.join(model_checkpoints_path, f"{config.config['2d_in_channels']}_in_channels")
  if not os.path.isdir(model_checkpoints_path):
    os.mkdir(model_checkpoints_path)

model_checkpoints_path = os.path.join(model_checkpoints_path, model_name) 
if not os.path.isdir(model_checkpoints_path):
  os.mkdir(model_checkpoints_path)



# %%
try:
  epoch = epoch
except NameError:
  epoch = 0

try:
  running_loss = running_loss
except NameError:
  running_loss = 0.0

try:
  epoch_loss = epoch_loss
except NameError:
  epoch_loss = []

epoch_0 = epoch + 1 if epoch else epoch

# %%
start_time = time.time()

if config.config['train']:
  model.train()

  for epoch in range(epoch_0, epoch_0 + NUM_EPOCHS):
    print(50*f"-")
    print(f"epoch: {epoch}/{epoch_0+NUM_EPOCHS-1}")

    random.shuffle(segments_id)
    for s_i, segment_id in enumerate(segments_id):
      print(f"segment {s_i}/{len(segments_id)-1} on epoch {epoch}/{epoch_0+NUM_EPOCHS-1}: {segment_id}")
      
      print("loading images...")
      image_stack = load_images(segment_id)

      mask, label = load_mask_label(segment_id)

      if segment_id in config.noink_labels:
        noink_label = load_noinklabel(segment_id)

        img = cv2.imread(os.path.join(NOINK_LABELS_PATH, f"{segment_id}.png"), 0)

        countours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        noink_x1y1x2y2s = []
        for cnt in countours:
          x1, y1, w, h = cv2.boundingRect(cnt)

          if w < 64 or h < 64:
            continue

          randx1 = x1 + random.randint(0, stride//2)
          randy1 = y1 + random.randint(0, stride//2)

          x1_list = list(range(randx1, x1+w-tile_size+1, stride))
          y1_list = list(range(randy1, y1+h-tile_size+1, stride))
          
          for y1 in y1_list:
            for x1 in x1_list:
              if mask[y1,x1] == False or noink_label[y1, x1] == False:
                continue
              y2 = y1 + tile_size
              x2 = x1 + tile_size

              inklabel = noink_label[y1:y2, x1:x2]
              ink_count = np.count_nonzero(inklabel)
              total_count = inklabel.size

              if ink_count/total_count > 0.85:
                noink_x1y1x2y2s.append((x1, y1, x2, y2))

        noink_x1y1x2y2s = np.array(noink_x1y1x2y2s)
        noink_mask_exists = True
      else:
        noink_mask = torch.empty(0)
        noink_mask_exists = False

      img = cv2.imread(os.path.join(LABELS_PATH,f"{segment_id}.png"), 0)

      countours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      x1y1x2y2s = []
      for cnt in countours:
        x1, y1, w, h = cv2.boundingRect(cnt)

        if w < 64 or h < 64:
          continue

        randx1 = x1 + random.randint(0, stride//2)
        randy1 = y1 + random.randint(0, stride//2)

        x1_list = list(range(randx1, x1+w-tile_size+1, stride))
        y1_list = list(range(randy1, y1+h-tile_size+1, stride))

        for y1 in y1_list:
          for x1 in x1_list:

            if mask[y1, x1] == False:
              continue

            if noink_mask_exists:
              if noink_label[y1, x1] == True:
                continue

            y2 = y1 + tile_size
            x2 = x1 + tile_size

            inklabel = label[y1:y2, x1:x2]
            ink_count = np.count_nonzero(inklabel)
            total_count = inklabel.size
            
            if ink_count/total_count > 0.7:
              x1y1x2y2s.append((x1, y1, x2, y2))
            
      x1y1x2y2s = np.array(x1y1x2y2s)
      
      if len(x1y1x2y2s) == 0:
        continue

      print(len(x1y1x2y2s),len(noink_x1y1x2y2s))

      random.shuffle(x1y1x2y2s)
      if noink_mask_exists and len(noink_x1y1x2y2s) > 0:
        random.shuffle(noink_x1y1x2y2s)
        len_noink = len(x1y1x2y2s) if len(x1y1x2y2s) < len(noink_x1y1x2y2s) else len(noink_x1y1x2y2s)
        len_ink = len(noink_x1y1x2y2s) if len(noink_x1y1x2y2s) < len(x1y1x2y2s) else len(x1y1x2y2s)
        
        ink_dataset = SubvolumeDataset(image_stack, torch.tensor([1.]).view(1), x1y1x2y2s[:len_ink], segment_id, transforms=get_transforms('training', training_aug_p[epoch]))
        noink_dataset = SubvolumeDataset(image_stack, torch.tensor([0.]).view(1), noink_x1y1x2y2s[:len_noink], segment_id, transforms=get_transforms('training', training_aug_p[epoch]))
        train_loader = data.DataLoader(data.ConcatDataset([ink_dataset, noink_dataset]), batch_size=BATCH_SIZE, shuffle=True, num_workers=6, drop_last=True)
      else:
        train_dataset = SubvolumeDataset(image_stack, torch.tensor([1.]).view(1), x1y1x2y2s, segment_id, transforms=get_transforms('training', training_aug_p[epoch]))
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, drop_last=True)
     
      print(len(train_loader))
      # del train_dataset
      # del image_stack

      print('training...')
      for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=len(train_loader)):

        optimizer.zero_grad()
        
        inklabels = inklabels.to(DEVICE)
        subvolumes = subvolumes.to(DEVICE)
        if config.config['2dmodel']:
          subvolumes = subvolumes.squeeze(1)

        # with torch.cuda.amp.autocast():
        outputs = model(subvolumes)
        loss = criterion(outputs, inklabels)
        # scaler.scale(loss).backward()
        loss.backward()
        running_loss += loss.detach().item()
        
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()

      print("loss:", running_loss / len(train_loader))
      running_loss = 0.0

    scheduler.step()

    if True or epoch % 5 == (5 - 1) and config.config['save_checkpoint']:
      torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        'running_loss': running_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
      }, os.path.join(model_checkpoints_path, f"{model_name}_backup_{epoch}.pth"))

  if config.config['save_checkpoint']:
        torch.save({
          'model_state_dict': model.state_dict(),
          'epoch': epoch,
          'epoch_loss': epoch_loss,
          'running_loss': running_loss,
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(model_checkpoints_path, f"{model_name}_{epoch}.pth"))

print(f'Training took {time.time() - start_time}')



# %% [markdown]
#   ## INFERENCE

# %%
segments_id = config.infering_segments_id


# %%
THRESHOLD = 0.5
sigmoid = nn.Sigmoid()


# %%
model_preds_path = './preds'
if not os.path.isdir(model_preds_path):
  os.mkdir(model_preds_path)

if config.config['3dmodel']:
  model_preds_path = os.path.join(model_preds_path, '3d')
  if not os.path.isdir(model_preds_path):
    os.mkdir(model_preds_path)

elif config.config['2dmodel']:
  model_preds_path = os.path.join(model_preds_path, '2d')
  if not os.path.isdir(model_preds_path):
    os.mkdir(model_preds_path)
  
  if config.config['2d_in_channels']:
    model_preds_path = os.path.join(model_preds_path, f"{config.config['2d_in_channels']}_in_channels")
  if not os.path.isdir(model_preds_path):
    os.mkdir(model_preds_path)

model_preds_path = os.path.join(model_preds_path, model_name) 
if not os.path.isdir(model_preds_path):
  os.mkdir(model_preds_path)

model_preds_path = os.path.join(model_preds_path, f"{model_name}_{epoch}")
if not os.path.isdir(model_preds_path):
  os.mkdir(model_preds_path)



# %%
start_time = time.time()

if config.config['infer']:
  model.eval()

  for s_i, segment_id in enumerate(segments_id):
    print(f"segment: {segment_id}")

    mask, label = load_mask_label(segment_id, load_label=False)

    x1_list = list(range(0, mask.shape[1]-tile_size+1, infer_stride))
    y1_list = list(range(0, mask.shape[0]-tile_size+1, infer_stride))

    x1y1x2y2s_mask = []
    for y1 in y1_list:
      for x1 in x1_list:
        y2 = y1 + tile_size
        x2 = x1 + tile_size

        if mask[y1,x1] == False:
          continue

        x1y1x2y2s_mask.append((x1, y1, x2, y2))

    x1y1x2y2s_mask = np.array(x1y1x2y2s_mask)

    print("loading images...")
    
    image_stack = load_images(segment_id)
    eval_dataset = SubvolumeDataset(image_stack, False, x1y1x2y2s_mask, segment_id, transforms=get_transforms('validation'))
    eval_loader = data.DataLoader(eval_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False, num_workers=6, drop_last=True)
    # del image_stack
    # del eval_dataset

    output = torch.zeros((label.shape[0], label.shape[1])).float().to(DEVICE)

    print(f'infering segment {segment_id} ({s_i}/{len(segments_id)-1})...')

    with torch.no_grad():
      for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):

        # with torch.cuda.amp.autocast():
        outputs = sigmoid(model(subvolumes.to(DEVICE)))
        for j, value in enumerate(outputs):
          x1, y1, x2, y2 = x1y1x2y2s_mask[i*INFER_BATCH_SIZE+j]
          output[y1:y2, x1:x2] += value.masked_fill_(value < THRESHOLD, 0) / (tile_size/infer_stride)

    img = (output*255).to(torch.uint8).cpu().numpy()
     
    Image.fromarray(img).save(os.path.join(model_preds_path,f'{segment_id}.png'))

print(f'Inference took {time.time() - start_time}')



