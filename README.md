# MMT: Multi-modal Transformer for Video Retrieval

![architecture](figs/Cross_mod_architecture.png)

## Intro

This repository provides the code for training our video retrieval cross-modal architecture.
Our approach is described in the paper "Multi-modal Transformer for Video Retrieval" [[arXiv](https://arxiv.org/abs/2007.10639), [webpage](http://thoth.inrialpes.fr/research/MMT/)]

Our proposed Multi-Modal Transformer (MMT) aggregates sequences of multi-modal features (e.g. appearance, motion, audio, OCR, etc.) from a video. It then embeds the aggregated multi-modal feature to a shared space with text for retrieval. It achieves state-of-the-art performance on MSRVTT, ActivityNet and LSMDC datasets.

## Installing
```bash
git clone https://github.com/gabeur/mmt.git
```

## Requirements
* Python 3.7
* Pytorch 1.4.0
* Transformers 3.1.0
* Numpy 1.18.1

```bash
cd mmt
# Install the requirements
pip install -r requirements.txt
```

## ECCV paper

In order to reproduce the results of our ECCV 2020 Spotlight paper, please first download the video features from [this page](http://thoth.inrialpes.fr/research/video-features/) by running the following commands:

```bash
# Create and move to mmt/data directory
mkdir data
cd data
# Download the video features
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/MSRVTT.tar.gz
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/activity-net.tar.gz
wget http://pascal.inrialpes.fr/data2/vgabeur/video-features/LSMDC.tar.gz
# Extract the video features
tar -xvf MSRVTT.tar.gz
tar -xvf activity-net.tar.gz
tar -xvf LSMDC.tar.gz
```

Download the checkpoints:
```bash
# Create and move to mmt/data/checkpoints directory
mkdir checkpoints
cd checkpoints
# Download checkpoints
wget http://pascal.inrialpes.fr/data2/vgabeur/mmt/data/checkpoints/HowTo100M_full_train.pth
wget http://pascal.inrialpes.fr/data2/vgabeur/mmt/data/checkpoints/MSRVTT_jsfusion_trainval.pth
wget http://pascal.inrialpes.fr/data2/vgabeur/mmt/data/checkpoints/prtrn_MSRVTT_jsfusion_trainval.pth
```

You can then run the following scripts:

### MSRVTT

#### Training from scratch

Training + evaluation:
```bash
python -m train --config configs_pub/eccv20/MSRVTT_jsfusion_trainval.json
```

Evaluation from checkpoint:
```bash
python -m train --config configs_pub/eccv20/MSRVTT_jsfusion_trainval.json --only_eval --load_checkpoint data/checkpoints/MSRVTT_jsfusion_trainval.pth
```

Expected results:
```
MSRVTT_jsfusion_test:
t2v_metrics/R1/final_eval: 24.1
t2v_metrics/R5/final_eval: 56.4
t2v_metrics/R10/final_eval: 69.6
t2v_metrics/R50/final_eval: 90.4
t2v_metrics/MedR/final_eval: 4.0
t2v_metrics/MeanR/final_eval: 25.797
t2v_metrics/geometric_mean_R1-R5-R10/final_eval: 45.56539387310681
v2t_metrics/R1/final_eval: 25.9
v2t_metrics/R5/final_eval: 58.1
v2t_metrics/R10/final_eval: 69.3
v2t_metrics/R50/final_eval: 90.8
v2t_metrics/MedR/final_eval: 4.0
v2t_metrics/MeanR/final_eval: 22.852
v2t_metrics/geometric_mean_R1-R5-R10/final_eval: 47.06915231647284
```

#### Finetuning from a HowTo100M pretrained model:

Training + evaluation:
```bash
python -m train --config configs_pub/eccv20/prtrn_MSRVTT_jsfusion_trainval.json --load_checkpoint data/checkpoints/HowTo100M_full_train.pth
```

Evaluation from checkpoint:
```bash
python -m train --config configs_pub/eccv20/prtrn_MSRVTT_jsfusion_trainval.json --only_eval --load_checkpoint data/checkpoints/prtrn_MSRVTT_jsfusion_trainval.pth
```

Expected results:
```
MSRVTT_jsfusion_test:
t2v_metrics/R1/final_eval: 25.8
t2v_metrics/R5/final_eval: 57.2
t2v_metrics/R10/final_eval: 69.3
t2v_metrics/R50/final_eval: 90.7
t2v_metrics/MedR/final_eval: 4.0
t2v_metrics/MeanR/final_eval: 22.355
t2v_metrics/geometric_mean_R1-R5-R10/final_eval: 46.76450299746546
v2t_metrics/R1/final_eval: 26.1
v2t_metrics/R5/final_eval: 57.8
v2t_metrics/R10/final_eval: 68.5
v2t_metrics/R50/final_eval: 90.6
v2t_metrics/MedR/final_eval: 4.0
v2t_metrics/MeanR/final_eval: 20.056
v2t_metrics/geometric_mean_R1-R5-R10/final_eval: 46.92665942024404
```

### ActivityNet

Training from scratch
```bash
python -m train --config configs_pub/eccv20/ActivityNet_val1_trainval.json
```

### LSMDC

Training from scratch
```bash
python -m train --config configs_pub/eccv20/LSMDC_full_trainval.json
```

## References
If you find this code useful or use the "s3d"(motion) video features, please consider citing:
```
@inproceedings{gabeur2020mmt,
    TITLE = {{Multi-modal Transformer for Video Retrieval}},
    AUTHOR = {Gabeur, Valentin and Sun, Chen and Alahari, Karteek and Schmid, Cordelia},
    BOOKTITLE = {{European Conference on Computer Vision (ECCV)}},
    YEAR = {2020}
}
```

The features "face", "ocr", "rgb"(appearance), "scene" and "speech" were extracted by the authors of [Collaborative Experts](https://github.com/albanie/collaborative-experts). If you use those features, please consider citing:
```
@inproceedings{Liu2019a,
    author = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
    booktitle = {British Machine Vision Conference},
    title = {Use What You Have: Video retrieval using representations from collaborative experts},
    date = {2019}
}
```

## Acknowledgements

Our code is structured following the [template](https://github.com/victoresque/pytorch-template) proposed by @victoresque. Our code is based on the implementation of [Collaborative Experts](https://github.com/albanie/collaborative-experts), [Transformers](https://github.com/huggingface/transformers) and [Mixture of Embedding Experts](https://github.com/antoine77340/Mixture-of-Embedding-Experts). We thank Maksim Dzabraev for discovering bugs in our implementation and notifying us of the issues (See the issues section for more detail).
