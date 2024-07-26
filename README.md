# TGTSF: Text Guided Time Series Forecasting

The main model is TGTSF_torch. Other versions are deprecated. 

We will update the paper on arxiv from time to time. Keep updated here: https://arxiv.org/abs/2405.13522


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

⚠ If you have trouble in downloading the pre-embedding files with git-lfs, we also provide google drive links for the pre-embedding files. [Click Here](https://drive.google.com/drive/folders/1feNhuijyls5DtxkeDjoKM0lISEF-UaPf?usp=sharing)

You can use `gdown` to download the files from google drive. 

## Run the Model

Run scripts in the ./scripts folder. 

Use visualize.ipynb to visualize the results. We may upload the checkpoint we trained later. 



