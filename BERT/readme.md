## Paper Name :- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Author :- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

URL - https://arxiv.org/abs/1810.04805

## Paper Summary
 BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task specific architecture modifications

 Two step framework - `pre-training` and `fine-tuning`

 During **pre-training**, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from thedownstream tasks.

### Model Architecture

BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

- BERT-BASE (L=12, H=768, A=12, Total Parameters=110M)
- BERT-LARGE (L=24, H=1024,A=16, Total Parameters=340M)

<img src="https://sushant-kumar.com/blog/bert-architecture.png">

Use WordPiece embeddings with a 30,000 token vocabulary. The first token of every sequence is always special classification token ([CLS]). Two sentence seperated with a special token ([SEP]). 

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-30_at_12.57.20_PM.png">


### Pre-training BERT
**Task #1: Masked LM** - Here we simply mask some parcentage(15%) of input token randomly and predict those masked token. Purpose - **understand the context of words in a sequence**


**Task #2: Next Sentence Prediction (NSP)** -  In order to understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Purpose - **understand the relationships between sentences**