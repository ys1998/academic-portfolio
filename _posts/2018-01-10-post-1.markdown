---
layout: post
title:  Title of post 1
date: 2018-01-10 15:10:00
description: An apt description of the post
categories: [rnn, discussion]
---

Write your post content here in normal `markdown`. An example post is shown below for reference. 

### Introduction

Recurrent Neural Networks and their variations are very likely to overfit the training data. This is due to the large network formed by unfolding each cell of the RNN, and *relatively* small number of parameters (since they are shared over each time step) and training data. Thus, the perplexities obtained on the test data are often quite larger than expected. Several attempts have been made to minimize this problem using varied **regularization** techniques. This paper tackles this issue by proposing a model that combines several of such existing methods.

*Merity et al*'s model is a modification of the standard **LSTM** in which *DropConnect* is applied to the hidden weights in the *recurrent* connections of the LSTM for regularization. The dropout mask for each weight is preserved and the same mask is used across all time steps, thereby adding negligible computation overhead. Apart from this, several other techniques have been incorporated :
* **Variational dropout** : The same dropout mask is used for a particular recurrent connection in both the forward and backward pass for all time steps. Each input of a mini-batch has a separate dropout mask, which ensures that the regularizing effect due to it isn't identical across different inputs.
* **Embedding dropout** : Dropout with dropout probability $$p_e$$ is applied to word embedding vectors, which results in new word vectors which are identically zero for the dropped words. The remaining word vectors are scaled by $$\frac{1}{1-p_e}$$ as compensation.
* **AR and TAR** : AR (Activation Regularization) and TAR (Temporal Activation Regularization) are modifications of $$L_2$$ regularization, wherein the standard technique is applied to dropped *output activations* and dropped *change in output activations* respectively. Mathematically, the additional terms in the cost function $$J$$ are (here $$\alpha$$ and $$\beta$$ are scaling constants and $$\textbf{D}$$ is the dropout mask) :

$$
J_{AR}=\alpha L_2\left(\textbf{D}_l^t\odot h_l^t\right)\\
J_{TAR}=\beta L_2\left(\textbf{D}_l^t\odot\left(h_l^t - h_l^{t-1}\right)\right)
$$

* **Weight tying** : In this method, the parameters for word embeddings and the final output layer are shared.
* **Variable backpropagation steps** : A random number of BPTT steps are taken instead of a fixed number, whose mean is very close to the original fixed value ($$s$$). The BPTT step-size ($$x$$) is drawn from the following distribution (here $$\mathcal{N}$$ is the Gaussian distribution, $$p$$ is a number close to 0.95 and $$\sigma^2$$ is the desired variance) :

$$
x \sim p\cdot \mathcal{N}\left(s,\sigma^2\right) + (1-p)\cdot \mathcal{N}\left(\frac{s}{2},\sigma^2\right)
$$

* **Independent sizes of word embeddings and hidden layer** : The sizes of the hidden layer and word embeddings are kept independent of each other.

The paper also introduces a new optimization algorithm, namely **Non-monotonically Triggered Averaged Stochastic Gradient Descent** or NT-ASGD, which can be programmatically described as follows :
{% highlight python %}
def NT_ASGD(f, t, w0, n, L, lr):
    """
    Input parameters :
    f  - objective function
    t  - stopping criterion
    w0 - initial parameters
    n  - non-monotonicity interval
    L  - number of epochs after which finetuning is done
    lr - learning rate

    Returns :
    parameter(s) that minimize `f`
    """
    k = 0; T = 0; t = 0; params = []; logs = []
    w = w0
    while t(w):
        # `func_grad` computes gradient of `f` at `w`
        w = w - lr * func_grad(f, w)
        params.append(w)
        k += 1
        if k%L == 0:
            # Compute model's perplexity for current parameters
            v = perplexity(w)
            if t > n and v > min(logs[t-n:t+1]):
                T = k
            logs.append(v)
            t += 1
    # Return the average of best `k-T+1` parameters
    return sum(params[-(k-T+1):])/(k-T+1)     
{% endhighlight %}
They also combined their **AWD-LSTM** (ASGD Weight Dropped LSTM) with a neural cache model to obtain further reduction in perplexities. A *neural cache model* stores previous states in memory, and predicts the output obtained by a *convex combination* of the output using stored states and the AWD-LSTM.

### Network description
*Merity et al*'s model used a 3-layer weight dropped LSTM with dropout probability `0.5` for **PTB corpus** and `0.65` for **WikiText-2**, combined with several of the above regularization techniques. The different hyperparameters (as referred to in the discussion above) are as follows : hidden layer size ($$H$$) = `1150`, embedding size ($$D$$) = `400`, number of epochs = `750`, $$L$$ = `1`, $$n$$ = `5`, learning rate = `30`, Gradients clipped at `0.25`, $$p$$ = `0.95`, $$s$$ = `70`, $$\sigma^2$$ = `5`, $$\alpha$$ = `2`, $$\beta$$ = `1`, dropout probabilities for input, hidden outputs, final output and embeddings as `0.4`, `0.3`, `0.4` and `0.1` respectively.

Word embedding weights were initialized from $$\mathcal{U}\left[-0.1,0.1\right]$$ and all other hidden weights from $$\mathcal{U}\left[-\frac{1}{\sqrt{1150}},\frac{1}{\sqrt{1150}}\right]$$. Mini-batch size of `40` was used for PTB and `80` for WT-2.

### Result highlights
* 3-layer AWD-LSTM with weight tying attained 57.3 PPL on PTB
* 3-layer AWD-LSTM with weight tying and a continuous cache pointer attained 52.8 PPL on PTB
