# Representation Learning of Daily Movement Data Using Text Encoders

This is the code used to produce the results presented in **[Representation Learning of Daily Movement Data Using Text Encoders](https://arxiv.org/abs/2405.04494)**.

## Abstract

Time-series representation learning is a key area of research for remote healthcare monitoring applications. In this work, we focus on a dataset of recordings of in-home activity from people living with Dementia. We design a representation learning method based on converting activity to text strings that can be encoded using a language model fine-tuned to transform data from the same participants within a $30$-day window to similar embeddings in the vector space. This allows for clustering and vector searching over participants and days, and the identification of activity deviations to aid with personalised delivery of care.

## This Repository

This repository contains the code to reproduce the results in the paper. The code is written in Python and uses `PyTorch` and HuggingFace's `Transformers` library as well as the `sentence-transformers` library.

Due to the sensitive nature of the dataset, we have not made the data or notebook publicly available. However, we provide the code required to fine-tune the language model and the code used to pre-process the dataset. These scripts will not run without the data, but we provide them for transparency and reproducibility.

The requirements for the code are provided in the `requirements.txt` file. Specifically, the code was run using Python 3.11.5 and the following packages:

```
numpy==1.26.1
pandas==2.1.2
torch==2.1.0
transformers==4.34.1
sentence-transformers==2.5.1
```

If you find this code helpful, please cite:
```
@article{capstick2024representation,
	title = {Representation Learning of Daily Movement Data Using Text Encoders},
	author = {Alexander Capstick and Tianyu Cui and Yu Chen and Payam Barnaghi},
	year = 2024,
	journal = {ICLR Workshop on Learning from Time Series For Health},
	url = {https://arxiv.org/abs/2405.04494}
}
```
