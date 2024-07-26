# TGTSF: Text Guided Time Series Forecasting

We only provide the toy dataset for the demo. We are working on uploading the rest of datasets. Especially the weather-large. 

We will update the link here: TODO

You may need to do the pre-embedding using scripts provided in the dataset folder. 

The main model is TGTSF_torch. Other versions are deprecated. 

We will update the paper on arxiv from time to time. Keep updated here: https://arxiv.org/abs/2405.13522

Updating the Dataset!~~~~


## Dataset Preparation

### Toy Dataset

- We have upload the toy dataset together with its generation scripts. You can use it to create your own dataset. The pre-embedding scripts are also included. Please do pre-embedding before training.

### Weather-Captioned Dataset

- Weather-captioned dataset is uploaded, including the time series of 10 years, all pre-embeddings files for the captions and hashtables for indexing the embeddings. 
- We put the pre-embeddings as tarball and storage them on github with git-lfs. You may need to:
  1. Install git-lfs with `sudo apt-get install git-lfs` or `brew install git-lfs`
  2. Run `git lfs install` in the repository
  3. Run `git lfs pull` to download the pre-embedding files.
  4. Unzip the tarball with `tar -xvf embeddings.tar`
  5. We break the embeddings for weather-large into several parts due to the 2GB file size limit of GITHUB. You need to merge them with `cat openai_caption_emb_large_part_*.tar > openai_caption_emb_large.tar` and then untar it.
- We also upload all the scripts to generate such a dataset, including rawdata, captioning, pre-embedding, and indexing as a seperate repository. You can find it here: [Weather Captioned Dataset](https://github.com/VEWOXIC/Weather-Captioned)

