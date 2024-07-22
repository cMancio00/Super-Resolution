# Super-Resolution

The aim image a super-resolution is to reconstruct a high-resolution image from a single low resolution image.

We will use the architecture presentend in:
[Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921) (Lim et all 2017).

# Deep Neural Network.

Generally speaking, a deep neaural network is an extention of a multilayer perceptron that has "many" hidden **layers**.

We can define a **layer** as a function:

$$\underbar{y} = \sigma \left( \mathbf{W}\underbar{x} + \underbar{b}  \right)$$

where:

- $\underbar{y} \in \mathbf{R}^K$
- $\mathbf{W} \in \mathbf{R}^{N \times D}$ the *wights matrix*
-  $\underbar{x} \in \mathbf{R}^D$ the *data*
- $\underbar{b} \in \mathbf{R}^N$ the *bias*
- $\sigma$ the *activation function* (applied element wise).

$N$ is the number of neurons in a given layer, and $D$ is the dimention of the input data.

As long as the dimension of a layer output matches the dimensional input of the next layer, we can connect them.

> [!NOTE]
> Just the data and the activation function is known. Bias and weights must be learned.

Depending on the problem we face, we have to choose an architecture and an adeguate activation function.



 