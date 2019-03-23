---
layout: post
title: Title of post 2
date: 2018-01-10 15:01:00
description: An apt description of your post
categories: [rnn, regularization]
---

Write your post content here in normal `markdown`. An example post is shown below for reference. 
 
### Introduction
Regularization is an important step during the training of neural networks. It helps to generalize the model by reducing the possibility of overfitting the training data. There are several types of regularization techniques, with L2 , L1, elastic-net (linear combination of L2 and L1 regularization), dropout and drop-connect being the major ones. While L2, L1 and elastic-net regularization techniques work by constraining the trainable parameters (or *weights*) from attaining large values (so that no drastic changes in output are observed for slight changes in the input; or in other words, they prefer *diffused* weights rather than *peaked* ones), **dropout** and drop-connect work by averaging the task over a dynamically and randomly generated large *ensemble* of networks. These networks are obtained by randomly disconnecting neurons (*dropout*) or weights (*drop-connect*) from the original network so as to obtain a subnetwork on which the training process is carried out (although for $$\approx1$$ epoch for each, since the chance of the same subnetwork being generated again is very rare).

Application of dropout to feedforward neural networks gives promising results. RNNs are thought of as individual *cells* that are *unfolded* over several time-steps, with the input at each time-step being a token of the sequence. When dropout is used for regularizing such a network, the 'disturbance' it generates at each time step propagates over a long interval, thereby decreasing the network's ability to represent long range dependencies. Thus, applying dropout in the standard manner to RNNs fails to give any improvement. It is here where *Zaremba et al*'s research comes into the picture.
### Network description
The network architecture that *Zaremba et al* proposed is quite simple and intuitive. In case of deep RNNs (i.e. RNNs spanning over several layers ($$h$$) where output of $$h_{l-1}^{t}$$ is used as the input for $$h_l^t$$), all connections between the cells in unfolded state can be broadly classified into two categories - *recurrent* and *non-recurrent*. The connections between cells in the same layer i.e. $$h_l^t ~-~ h_l^{t+1}~~\forall t$$ are *recurrent* connections, and those between cells in adjacent layers i.e. $$h_l^t ~-~ h_{l+1}^t~~\forall l$$ are *non-recurrent* connections. *Zaremba et al* suggested that **dropout** should be applied only to non-recurrent connections - thereby preventing problems which arose earlier.

The modified network for LSTM units can be mathematically represented as follows:

Denoting $$T_{m,n}$$ as an affine transformation from $$\mathbb{R}^m\rightarrow\mathbb{R}^n$$ ( i.e. $$T_{m,n}(x)=Wx+b$$ where $$x\in\mathbb{R}^{m\times1}$$, $$W\in\mathbb{R}^{n\times m}$$ and $$b\in\mathbb{R}^{n\times1}$$ and similarly for multiple inputs) and $$\otimes$$ as elementwise multiplication, we have :

$$
f_l^t=\text{sigmoid}\left(T_{N,D}^1\left(\textbf{D}\left(h_{l-1}^t\right) ; h_l^{t-1}\right)\right) \\
i_l^t=\text{sigmoid}\left(T_{N,D}^2\left(\textbf{D}\left(h_{l-1}^t\right) ; h_l^{t-1}\right)\right) \\
o_l^t=\text{sigmoid}\left(T_{N,D}^3\left(\textbf{D}\left(h_{l-1}^t\right) ; h_l^{t-1}\right)\right) \\
u_l^t=\text{tanh}\left(T_{N,D}^4\left(\textbf{D}\left(h_{l-1}^t\right) ; h_l^{t-1}\right)\right) \\
c_l^t=c_l^{t-1}\otimes f_l^t + u_l^t\otimes i_l^t \\
h_l^t = \text{tanh}\left(c_l^t\right)\otimes o_l^t
$$

Here, $$\textbf{D}$$ is the dropout *layer* or operator which sets a subset of its input randomly to zero with dropout probability `p`. This modification can be adopted for any other RNN architecture.

*Zaremba et al* used these architectures for their experiments in which each cell was unrolled for 35 steps. Mini-batch size was 20 for both :

**Medium LSTM** :
Hidden-layer dimension = `650`,
Weights initialized uniformly in `[-0.05,0.05]`,
Dropout probability = `0.5`,
Number of epochs = `39`,
Learning rate = `1` which decays by a factor of `1.2` after 6 epochs,
Gradients clipped at `5`.

**Large LSTM** :
Hidden-layer dimension = `1500`,
Weights initialized uniformly in `[-0.04,0.04]`,
Dropout probability = `0.65`,
Number of epochs = `55`,
Learning rate = `1` which decays by a factor of `1.15` after 14 epochs,
Gradients clipped at `10`.
### Result highlights
* 78.4 PPL on Penn TreeBank dataset using a single **Large LSTM**
* 68.7 PPL on Penn TreeBank dataset using an ensemble of 38 **Large LSTM**s
