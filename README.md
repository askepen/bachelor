# Phase-aware decompression of speech data using deep learning

## Requirements
Only Linux or MacOS are currently supported, as we are using the `sox` python library for audio processing, which is not available for Windows.

## Getting started
To get started, we recommend you create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:
```
pip install -r requirements.txt
```

### Train the model
When the requirements are installed, the model can be trained by running
```
python src/train.py [--hparam ...]
```
where hparams are a combination of the following:

| hparam | default | explanation |
|-|-|-|
| batch_size | 16 | number  | 
| lr | 1e-3 | Learning rate |

