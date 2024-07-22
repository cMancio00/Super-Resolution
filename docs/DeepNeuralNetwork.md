# Deep Neural Network

Generally speaking, a deep neaural network is an extention of a multilayer perceptron that has "many" hidden **layers**.

We can define a **layer** as a function:

$$\underline{y} = \sigma \left( \mathbf{W}\underline{x} + \underline{b}  \right)$$

where:

- $\underline{y} \in \mathbf{R}^K$
- $\mathbf{W} \in \mathbf{R}^{N \times D}$ the *wights matrix*
-  $\underline{x} \in \mathbf{R}^D$ the *data*
- $\underline{b} \in \mathbf{R}^N$ the *bias*
- $\sigma$ the *activation function* (applied element wise).

$N$ is the number of neurons in a given layer, and $D$ is the dimention of the input data.

As long as the dimension of a layer output matches the dimensional input of the next layer, we can connect them.

> [!NOTE]
> Just the data and the activation function is known. Bias and weights must be learned.

Depending on the problem we face, we have to choose an architecture and an adeguate activation function.

A deep Network will have the following structure depending on the task, Regression or Classification.

![Deep Neural Network](/docs/Images/NeuralNetwork.png)

> [!CAUTION]
> Regression tasks will require differents strategies than Classification ones.
> Concepts and methods remain the same.

# How to build a Deep Neural Network

The building process can be splitted in these three areas.

- Design
- Train
- Regularize

> [!NOTE]
> These areas are application domain agnostic

## Design

First thing first, we have to identify the task we want to solve. Is it a Regrssion problem or a Classification one?

Check if already exist an architecture for that kind of problem, if not, extend an existing one.

Given the data, choose an **activation function** that best fit the task.

Choose also the activation function of the **output layer**. As the name suggest, it is the last layer of the network and the output of the network.

Now that we have the architecture composed of:

- Number of neurons per layer
- Number of layers (depth)
- Activation function (for each layer)
- Output function

We can think of training strategies.

## Train

The training is an optimization problem.

An unconstrained optimization problem can be described as follows:

$$\min_{x} f(x)$$

The goal is to minimize $f(x)$.
Our function will be a **loss function**, which is a mesurement of the error of our prediction with respect to the real value. The loss function will depend on the task we want to solve.

The framework to find a minimum, given a strating point $x_0$ is:

$$x_{k+1} = x_{k} + \alpha_{k}d_{k}$$

We choose a direction $d_{k}$ and a step $\alpha_{k}$ from $x_{k}$ to get the new point $x_{k+1}$. We repet these steps until convergence or until some stopping criteria is met.

In a machine learning prespective $\alpha$ is called **learning rate** and can be choosen with algorithms like **ADAM**.

The direction is the **anti-gradient** $-\nabla{f(x)}$ which is the steepest descent direction.

The **loss** is usually a convex function that garantees the existance of a minimum on the function. However we are not interest in the minimum but in a good aproximation of it to avoid overfitting and increase generalisation.

The gradient method has sublinear iteration complexity, however, if the function is **strongly convex** the iteration complexity is **linear**.

A strongley convex function is defined as:

$$f(x) = g(x) + \mu||x||^2$$

where $g(x)$ is convex and $\mu||x||^2$ is a regularization term.

So we can speed up computation and traing minimizing a stronglly convex function.

How can we calculate the loss of a complex function?

We need a way to *move* from the input to the output layer which is called **forward**. We are then able to calculate the loss comparing the result with real data. We then calculate the gradiend with the **back propagation** algorithm that exploit the chain rule and the **computational graph** (usally build by the framework).

Given the gradiend we can update the weights, and iterate. Each iteration is called **epoch**.

## Regularize

These are tecniques used to avoid overfitting.

They include but not limited to:

1. Early stopping the training
2. Adaptive learning rate (Adam handles it)
3. Data augmentation