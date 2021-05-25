# Adding codes in progress

# Sentiment Analysis (SA) using Deep Learning-based language representation learning models

## Introducation (English)
<p align="justify">
Deep learning (DL) approaches use various processing layers to learn hierarchical representations of data. Recently, many methods and designs of natural language processing (NLP) models have shown significant development, especially in text mining and analysis. For learning vector-space representations of text, there are famous models like Word2vec, GloVe, and fastText. In fact, NLP took a big step forward when BERT and recently GTP-3 came out.
Deep Learning algorithms are unable to deal with textual data in their natural language data form which is typically unstructured information; they require special representation of data as inputs instead. Usually, natural language text data needs to be converted into internal representations form that DL algorithms can read such as feature vectors, hence the necessity to use representation learning models. These models have shown a big leap during the last years. Their set ranges from the methods that embed words into distributed representations and use the language modeling objective to adjust them as model parameters (like Word2vec, fastText, and GloVe), to recently transfer learning models (like ELMo, BERT, ULMFiT, XLNet, and GPT-2). These last use larger corpora, more parameters, more computing resources, and instead of assigning each word with a fixed vector, they use multilayer neural networks to calculate dynamic representations for the words according to their context, which is especially useful for the words with multiple meanings.</p>

## Introducation (Francais)
<p align="justify">Le traitement automatique du langage naturel (NLP=Natural Language Processing) vise à convertir le langage humain en une représentation formelle en utilisant différentes techniques de calcul. Ce domaine progresse rapidement en raison de l'intérêt croissant pour les communications homme-machine, de la grande quantité de données textuelles stockées sur le Web, des puissants systèmes informatiques et des algorithmes améliorés. En effet, les algorithmes et architectures d'apprentissage en profondeur (DL=Deep Learning) ont fait des progrès remarquables ces dernières années dans les domaines de l'analyse de texte, cependant il nécessite une représentation spéciale des données en tant qu'entrées. 
À cet égard, les modèles d'apprentissage de la représentation du langage ont fait un grand saut au cours des dernières années, allant de méthodes qui incorporent les mots dans des représentations distribuées et utilisent l'objectif de modélisation du langage pour les ajuster en tant que paramètres de modèle comme Word2vec, GloVe, et fastText, à l'apparition de modèles d'apprentissage par transfert comme BERT, ELMo, ULMFiT, GPT-2 et XLNet qui utilisent des corpus plus grands, plus de paramètres, plus de ressources informatiques, et au lieu d’affecter à chaque mot un vecteur fixe, ils utilisent des réseaux de neurones multicouches pour calculer les représentations dynamiques des mots en fonction de leur contexte.</p>

## Language Representation Learning Models
<p align="justify">One of the important tasks in NLP is the learning of vector representations of text, as deep learning algorithms require representing their input entries in a vector format.</p>

#### Neural Word Embeddings
<p align="justify"><b>- <a href="https://arxiv.org/abs/1301.3781">Word2vec</a></b> is an unsupervised learning algorithm that consists of a group of related models used for word embeddings generation. It is based on three-layer neural networks and seeks to learn the vector representations of words composing a text, so that words that share similar contexts are represented by close digital vectors. Word2Vec has two neural architectures, called Continuous Bag-of-Words (CBOW) and Skip-Gram. CBOW receives as input the context of a word, i.e., the terms surrounding it in a sentence, and tries to predict the word in question. Skip-Gram does exactly the opposite: it takes a word as input and tries to predict its context.</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1607.04606">fastText</a></b> is a Facebook's AI library for efficient learning of sentences classification and word embeddings. It supports multiprocessing during training and allows to create an unsupervised or supervised learning algorithm to obtain vector representations of words and sentences. fastText uses a neural network for word embeddings and supports training continuous bag of words (CBOW) or skip-gram model. It can be used as an initializer for transfer learning.</p>

<p align="justify"><b>- <a href="https://www.aclweb.org/anthology/D14-1162">Glove</a></b> is an unsupervised learning algorithm to obtain word vector representations. This is accomplished by mapping words in a meaningful space where the distance between words is related to semantic resemblance. Training is performed using an underlying count-based model on the aggregated global word to word co-occurrence matrix within a text corpus, and the subsequent representations display interesting linear substructures in the word vector space. It combines the features of two sets of models, namely the local context window approaches and the global matrix factorization.</p>

#### Transfer Learning Techniques
<p align="justify"><b>- <a href="https://arxiv.org/abs/1802.05365">ELMo (Embeddings from Language Models)</a></b> is a pre-trained biLSTM (bidirectional LSTM) language model. Word embeddings is calculated by taking a weighted score of the hidden states from each layer of the LSTM. Weights are learned with downstream model parameters for a particular task, but LSTM layers are kept constant. Thus, the same word under different contexts can have different word vectors.</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1810.04805">BERT (Bidirectional Encoder Representations from Transformers)</a></b> is another language representation learning model that uses an attention transformers mechanism to learn the contextual relations between words in a text instead of bidirectional LSTMs to encode context which shows that pre-training transformer networks on a masked language modeling objective leads to even better performance by precisely adjusting the transformer weights  over a wide range  of NLP tasks.</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1907.11692">RoBERTa (A Robustly Optimized BERT Pretraining Approach)</a></b> is an optimized model resulting from analysis of Google's BERT training model and the identification of several changes to the training procedure that enhance its performance by Facebook AI and the University of Washington researchers. Specifically, these researchers used a novel and bigger dataset for training, trained the model over far more epochs, and removed the next sequence prediction training objective.</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1909.11942">ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)</a></b> is a “Lite” version of BERT, this model architecture includes two parameter-reduction methods: cross-layer parameter sharing and factorized embeddings parameterization. Furthermore, the proposed method contains a self-supervised loss for the sentence-order prediction.</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1801.06146">ULMFiT (Universal Language Model Fine-Tuning for Text Classification)</a></b> is a transfer learning method that can be applied to NLP. It uses a regular 3-layer LSTM architecture for either pre-training and fine-tuning tasks. ULMFiT consists of three steps: General-domain language model (LM) pertaining (pertaining language model on a large general-domain corpus), target task LM fine-tuning (the LM fine-tuned on the data of the target task), and the target task classifier fine-tuning (fine-tuning the classifier).</p>

<p align="justify"><b>- <a href="https://arxiv.org/abs/1906.08237">XLNet</a></b> is a generalized autoregressive (AR) pertaining method that uses the context word to predict the next word which is constrained to a unidirectional context, either backward or forward. Although, XLNet learns from bidirectional context using Permutation Language Modeling. It also influences the best of both AR language modeling and autoencoders while avoiding their limitations.</p>

<p align="justify"><b>- <a href="https://openai.com/blog/better-language-models">GPT-2 (Generative Pretrained Transformer 2 - successor of GPT)</a></b> follows the OpenAI GPT model with a few architecture modifications. It consists of a big transformer-based language model with 1.5 billion parameters, trained with the objective of the prediction of the next word, given all previous words in a text. And unlike the previous models that require pre-training and fine-tuning, there is no fine-tuning step for GPT-2.</p>

## Dataset
<p align="justify">Dataset used in experiments was combined from CARER-Emotion, DailyDialog, CrowdFlower, and Isear to create a rich dataset with 5 labels: anger (5k sentences), joy (26k sentences), sad (13k sentences), fear (3.6k sentences), and neutral (94k sentences). The used texts consist of tweets, dialog utterances, and short messages as shown in the table bellow.</p>

| Dataset | Year | Content | Number of sentences | Emotion categories |
| --- | --- | --- | --- | --- |
| [CARER – Emotion](https://www.aclweb.org/anthology/D18-1404) | 2018 | Tweets | 20k | Anger, anticipation, disgust, fear, joy, sadness, surprise, and trust |
| [DailyDialog](https://arxiv.org/abs/1710.03957) | 2017 | Dialogues | 102k | Neutral, joy, surprise, sadness, anger, disgust, and fear |
| [CrowdFlower](https://data.world/crowdflower/sentiment-analysis-in-text) | 2016 | Tweets | 40k | Empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, anger |
| [Isear](https://pubmed.ncbi.nlm.nih.gov/8195988/) | 1994 | Emotion situations | 7.5k | Joy, fear, anger, sadness, disgust, shame, guilt |

## Results

| Algorithm | Validation Accuracy | Validation Loss | Precision | Recall | F1-score | Training Time&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Number of parameters&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ***Word2vec*** | 0.8452 | 0.4178 | 0.8410 | 0.8453 | 0.8391 | Step: 63 ms<br/>Epoch: 222s<br/>Total: 2472s | Trainable: 440 581<br/>Non-Trainable: 11 060 100 |
| ***fastText*** | 0.8212 | 0.4908 | 0.8167 | 0.8223 | 0.8128 | Step: 88 ms<br/>Epoch: 311<br/>Total: 1567 | Trainable: 4 613<br/>Non-Trainable: 8 640 128 |
| ***Glove*** | 0.8391 | 0.4412 | 0.8362 | 0.8388 | 0.8297 | Step: 245 ms<br/>Epoch: 869s<br/>Total: 3576s | Trainable: 527 369<br/>Non-Trainable: 0 |
| ***ELMo*** | 0.8152 | 0.4810 | 0.8041 | 0.8102 | 0.8064 | Step: 7 ms<br/>Epoch: 844s<br/>Total: 3576s | Trainable: 527 369<br/>Non-Trainable: 0 |
| ***BERT*** | 0.8612 | 0.3551 | 0.8589 | 0.8612 | 0.8596 | Step: 787 ms<br/>Epoch: 247 min<br/>Total: 495 min | Trainable: 109 361 669<br/>Non-Trainable: 0 |
| ***RoBERTa*** | 0.8622 | 0.3629 | 0.8574 | 0.8533 | 0.8548 | Step: 609 ms<br/>Epoch: 192 min<br/>Total: 579 min | Trainable: 125 240 069<br/>Non-Trainable: 00 |
| ***ALBERT*** | 0.8558 | 0.3845 | 0.8514 | 0.8537 | 0.8468 | Step: 595 ms<br/>Epoch: 188 min<br/>Total: 567 min | Trainable: 11 687 429<br/>Non-Trainable: 0 |
| ***ULMFiT*** | 0.8509 | 0.4315 | 0.8472 | 0.8509 | 0.8476 | Step: -<br/>Epoch: 192s<br/>Total: 1920s | Trainable: 62 805<br/>Non-Trainable: 0 |
| ***XLNet*** | 0.8574 | 0.3697 | 0.8562 | 0.8583 | 0.8564 | Step: 1s<br/>Epoch: 430 min<br/>Total: 1293 min | Trainable: 117 312 773<br/>Non-Trainable: 0 |
| ***GPT-2*** | 0.8591 | 0.3796 | 0.8559 | 0.8591 | 0.8549 | Batch: 67 ms<br/>Epoch: 18 min<br/>Total: 73 min | Trainable: 124 439 808<br/>Non-Trainable: 0 |

## Discussion (English)
<p align="justify">In this work, we show the application of deep learning-based language representation learning models for the classification of 5 sentiment types based on a combined dataset. We notice that transfer learning approaches reach the best average results using the training and validation data in fewer epochs than word embeddings ones, because it benefits from other base models’ knowledge. Nevertheless, it takes more time to train, due to the huge number of parameters used. Among these transfer learning approaches, we conclude that the best one is BERT algorithm because it reaches the best results in almost all our metrics, with 35.51% as validation loss, 85.89% as precision, 86.12% as recall, and 85.96% as F1-score in 495 min (2 epochs). For the accuracy, RoBERTa model has the best accuracy, with 86.22% in 579 min (3 epochs). On the other hand, transformer-based techniques reach their best result in more time (more than one hour to be trained) compared to the other models.</p>

<p align="justify">By examining these results, it is clear that BERT model performed the best results compared to the other methods, since it takes everything into account, in order to predict the true meaning of sentences. This means that transfer learning algorithms can achieve better classification results and learn additional correlations, but in terms of computation time, it consumes more because more parameters are needed as shown in Table 3. In fact, most DL architectures use similar computational elements; therefore, it is a convention to use the number of parameters as a stand-in for complexity, although those networks may have the same number of parameters but require different numbers of operations (ALBERT for example is configured to share all parameters including feed-forward network and attention parameters across layers).</p>

<p align="justify">Despite all efforts, our models tend to overfit. In fact, models trained on text data are subject to overfitting due to the use of OOV (Out of Vocabulary) token in NLP-based models. OOV is used to handle unseen words. There is a high chance of unseen words in NLP models, and overfitting occurs when the model is trained heavily on the training data but cannot generalize well to unseen data. Those unseen words generate a scenario where the model is strongly tuned to the training set. Hence, we stop at the epoch when each algorithm begins to over-fit.</p>

## Discussion (Francais)
<p align="justify">Dans ce travail, nous montrons l'application de modèles d'apprentissage de la représentation linguistique basés sur l'apprentissage profond pour la classification de 5 types de sentiments en utilisant un ensemble de données combiné. Nous avons observé que les approches d'apprentissage par transfert atteignent les meilleurs résultats en utilisant les données d’entrainement et de validation et avec moins d'époques que celles par intégration de mots, car elles bénéficient des connaissances d’autres modèles, mais il leurs faut plus de temps pour s'entraîner en raison du très grand nombre de paramètres.</p>

<p align="justify">Parmi ces approches d'apprentissage par transfert, nous avons constater que le meilleur est l'algorithme BERT car il atteint les meilleurs résultats dans presque toutes nos métriques, avec 35,51% de perte, 85,89% d’exactitude, 86,12% de rappel et 85,96% de F1-score en 495 min (2 époques). Pour la précision, le modèle RoBERTa est le meilleur, avec 86,22% en 579 min (3 époques). En revanche, les techniques basées sur les Transformer atteignent leur meilleur résultat mais en plus de temps (plus d'une heure pour s’entrainer) par rapport aux autres modèles.</p>

<p align="justify">Le modèle BERT a donné des meilleurs résultats par rapport aux autres méthodes, car il prend tout en compte, afin de prédire le vrai sens des phrases. Cela signifie que les algorithmes d'apprentissage par transfert peuvent obtenir de meilleurs résultats de classification en apprenant des corrélations supplémentaires. Mais en temps de calcul, ils consomment plus parce que plus de paramètres sont nécessaires. En fait, la plupart des architectures DL utilisent des éléments de calcul similaires, par conséquent, on peut utiliser le nombre de paramètres comme substitut de la complexité, bien que ces réseaux puissent avoir le même nombre de paramètres mais nécessitent des nombres d'opérations différents.</p>

# Citation
Please cite the [following paper](http://ijeecs.iaescore.com/index.php/IJEECS/article/view/24400) when you can:

Veuillez citer l'[article suivant](http://ijeecs.iaescore.com/index.php/IJEECS/article/view/24400) lorsque vous le pouvez :

<pre><code>
@article{boukabous2021dl),
    title={A Comparative Study of DL-Based Language Representation Learning Models},
    author={Boukabous, Mohammed and Azizi, Mostafa},
    doi = {10.11591/ijeecs.v22.i2.pp424-432},
    issn = {24400-47768},
    journal = {IAES Indonesian Journal of Electrical Engineering and Computer Science (IJ-EECS)},
    keywords = {Natural Language Processing; Representation Models; BERT; GPT-2; Deep Learning; Sentiment Analysis},
    month = {may},
    url = {http://ijeecs.iaescore.com/index.php/IJEECS/article/view/24400},
    volume = {22},
    year = {2021}
}
</code></pre>
