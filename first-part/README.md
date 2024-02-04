## Model and Usage

The model used was [MedNeXt](https://arxiv.org/abs/2303.09975). The input and output shape for it was $32\times 64\times 64$. However, for classification, we modified the model's output to return a scalar value with:

```
return x.mean(4).mean(3).sum(dim=2)
```

To run it, 94G RAM and 24G GPU VRAM are enough. To setup, we can create a [```conda```](https://docs.anaconda.com/free/anaconda/install/linux/) environment. Then we create and activate an environment as:

```
conda update conda
conda create -n venv_vesuvius python=3.11
conda activate venv_vesuvius
```

Then, we can install the requirements:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
git clone https://github.com/erdpx/mednext-vesuvius-gp.git
cd mednext-vesuvius-gp
pip install -e .
cd ..
```

Now, inside the ```config.py``` file, we must change the PATH for the segments' directory, as well as the ink and no-ink labels, where it expects the following structure for each segment:
```
{segment_id}/
  layers/
    00.tif
    .
    .
    .
    65.tif
  {segment_id}_mask.png
```

It must be the same ```segment_id``` as appears in the ```segments_config.py```, where you choose the training and inferring segments. Checkpoint files are also changed in this file by changing its path and setting ```'load_checkpoint': True```. Then it can run with
```
python vesuvs.py
```
