# Cost-sensitive Regularization for Label Confusion-aware Event Detection

This is the source code for paper "Cost-sensitive Regularization for Label Confusion-aware Event Detection" in ACL2019.

## Requirements

* Tensorflow >= 1.2.0

## Usage
First, please unzip the word2vec embeddings in "word2vec_data/"

* gzip -d word2vec_data/ere2017.english.giga.lower.token.sg.win5.neg5.iter5.dim300.txt.gz
* gzip -d word2vec_data/word_word2vec.dat.gz

Then enter "src/CNN/src_*" dir, run the program like

* python DMCNN.py config_csr_instance.cfg

Hyperparameters in our paper are saved in configure files.

## Citation
Please cite:
* Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun. *Nugget Proposal Networks for Chinese Event Detection*. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.

```
@InProceedings{lin-Etal:2019:ACL2019nugget,
  author    = {Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le},
  title     = {Cost-sensitive Regularization for Label Confusion-aware Event Detection},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2019},
  publisher = {Association for Computational Linguistics}
}
```

## Contact
If you have any question or want to request for the data(only if you have the license from LDC), please contact me by
* hongyu2016@iscas.ac.cn
