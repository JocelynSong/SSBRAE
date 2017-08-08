# SSBRAE
Semantically Smooth Bilingual Phrase Embeddings for Statistic Machine Translation

Based on the BRAE, we extend the conventional bilingual recursive autoencoders by preserving the translation and the paraphrase probability distributions via regularization terms when learning bilingual phrase embeddings.
/exp/gbrae/brae/brae_batch.py is the function to train brae.
/exp/gbrae/gbrae/gbrae_batch.py is the function to train ssbrae. We can choose translation probability distributions or paraphrase probability distributions or both.
/src/rae_batch.py is our model.
