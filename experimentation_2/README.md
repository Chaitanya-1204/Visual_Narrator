# Visual_Narrator
Visual Narrator: End-to-End Image Captioning and Speech for the Visually Impaired

## Installation

First Create a environment for the project 

```bash 
conda create -n captioning python=3.10 -y
```

```bash
conda activate captioning
```

To install `torch`, `torchvision`, and `torchaudio` with **CUDA 12.1** support, run the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

All other requirements can be installed by running the following commands 

```bash
pip install -r requirements.txt
```


## Datasets 


For the iamge Captioning Part we are using MS COCO 2014 dataset 

* Training data has 82783 images each with 5 captions 

* Validation data has 40504 images each with 5 captions 

### Folder Structure

Organize the COCO dataset as follows:

```
coco/
├── annotations/
│   ├── captions_train2014.json
│   └── captions_val2014.json
├── train2014/
│   └── COCO_train2014_000000000000.jpg
└── val2014/
    └── COCO_val2014_000000000000.jpg
```



