# My Transformer

## Introduction

This is a Transformer I made based on *Attention is All You Need*. It implemented the transformer model. And it implemented the function of translating English into France.

## Usage

For trying the translator, you can download the source code and path to its directory. Then use `python train.py` command to train the model. The weights of Transformer will be stored in `./weights`. 

After training model, you can use `python translate.py`.  Follow the terminal instruction and input an English sentence to  be translated. Then the French version of the sentence can be seen on the terminal. 

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/translate.png)

## Detail

This part is about the architecture of my transformer.

### Transformer

Transformer is an Seq2Seq network. It mainly contains two part called *Encoder* and *Decoder*, as is depicted.

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/transformer.jpg)

### Encoder & Decoder

Encoder and Decoder is made of several EncoderLayer/DecoderLayer, including embedding, positional encoding and normalization. 

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/encoder%26decoder.jpg)

Embedding words has become standard practice in machine translation. When each word is fed into the network. Embedding will retrieve its vector. These vectors will then be learnt as parameters by the model.

The positional encoding will make the position of a word make sense to the model. In other words, position encoding tell the model where each word locate in a sentence. The position-specific values are determined by formulas:

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/pe.png)

The EncoderLayer and DecoderLayer is depicted as below.

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/encoder%26decoder%20layer.jpg)

It is not difficult to figure out that the main part is the layer called *attention* which, in this case, corresponds to *MultiHead-Attention*. *Attention mechanism* is the core of *Transformer*. 

### Attention mechanism

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the  query, keys, values, and output are all vectors. 

From my perspective, attention is can be described as *soft index*. It means that the output value is not a single value, but a weighted sum of all values, where the weight of each value is decided by the similarity between a single key and the given query. The similarity can be calculated by the distance function.

The scaled dot-product attention is calculated as below, where $Q$ is the query, $K$ is the key, $V$ is the value and $d_k$ is the dimension of key. 

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/attention.png)

MultiHead-Attention is based on this *Attention Mechanism*. It divides input into several heads before attention calculation. After that, the score results are concated altogether.

![](https://github.com/UniqueMR/Self-Transformer/blob/main/img/multihead-attention.jpg)
