# langcoach-mpl-mdd

### A project for Deep Learning course, this open source repo is used for training the model forked from [this repo](https://github.com/Mu-Y/mpl-mdd)

This repo contains code for fine-tuning a wav2vec2-based MDD model with momentum pseudo-labeling (MPL). The implementation is based on [SpeechBrain](https://github.com/speechbrain/speechbrain).

## Pull the repo
```
git clone git@github.com:Mu-Y/mpl-mdd.git
cd mpl-mdd
git submodule update --init --recursive
```

## Install dependencies and set up env
Install the requirements by SpeechBrain and some extras.
```
cd mpl-mdd/speechbrain
pip install -r requirements.txt
pip install textgrid transformers librosa
```
Append the path to speechbrain module to `PYTHONPATH`.
```
export PYTHONPATH=$PYTHONPATH:<path to mpl-mdd/speechbrain>
```

## Data preperation
First, download [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) dataset, and unzip it. Then run the following commands:
```
# for labeled samples - get train.json and test.json
python l2arctic_prepare.py <path to L2-ARCTIC>

# for unlabled samples - get train_unlabeled.json
python l2arctic_unlabeled_prepare.py <path to L2-ARCTIC>

# split dev set from training - get train-train.json and train-dev.json
python split_train_dev.py --in_json=data/train.json --out_json_train=data/train-train.json --out_json_dev=data/train-dev.json
```



## Training
### Step 1
Fine-tune a pre-trained wav2vec2 model on labeled samples.
```
python train.py hparams/train.yaml
```
### Step 2
Fine-tune the model from step 1 with momentum pseudo-labeling, using both labeled and unlabled samples.
```
python train_mpl.py hparams/train_mpl.yaml
```

## Evaluate the trained model
```
python evaluate.py hparams/evaluate.yaml
```
This will print PER and MDD F1, and write the PER and MDD details files.

## Inference with the trained model
```
python transcribe.py hparams/transcribe.yaml
```
By default, this command will write predictions of L2-ARCTIC test set into a json file. You can change the save path in `hparams/transcribe.yaml`.

## Acknowledgements
This code is adapted from:
https://github.com/Mu-Y/mpl-mdd

