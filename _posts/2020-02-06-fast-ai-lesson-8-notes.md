---
layout: post
title: Fast.ai Lesson 8 Notes
date: 2020-02-06
tags: deep-learning machine-learning fastai
description: My personal notes on Lesson 8 of part 2 of fast.ai (2019) -- <b>Matrix Multiplication; Forward and Backward Passes</b>. 
featured_image: fastai/image-20190706182251357.png
comments: true
---


## Overview

Part 2 of FastAI 2019 is 'bottom-up' - building the core of the FastAI library from scratch using PyTorch.

This lesson implements matrix multiplication in pure Python, then refactors and optimizes it using broadcasting and einstein summation. Then this lesson starts to look at the initialization of neural networks. Finally the lesson covers handcoding the forward and backwards passes of a simple model with linear layers and ReLU, before refactoring the code to be more flexible and concise so that you can understand how PyTorch's work.

Lesson 8 [lecture video](https://course.fast.ai/videos/?lesson=8).

<div class="row">
<div class="col-md-6" id="mdtoc">

__Table of Contents__

* TOC
{:toc}
</div>

</div>

## Different Matrix Multiplication Implementations

### Naive Matmul

A baseline **naive implementation** in pure python code:

```python
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c
```

Time: __3.26s__

_Doing loops in pure python and updating array elements one at a time is anathema to performance in python_.



### Elementwise Matmul

```python
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
```

Time: __4.84ms__

_The loop over `k` is replaced with a `sum()` over the elements of row slice in `a` times the column slice in `b`. This operation is outsourced to library calls in numpy which are likely compiled code written in C or Fortran, which gives the near 1000x speed-up._



### Broadcasting matmul

```python
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c
```

Time: __1.11ms__

_WTH is this? As is almost always the case, optimizing code comes at the expense of code readability. Let's work through this to convince ourselves that this is indeed doing a matmul._



#### Aside: Proof of Broadcasting Matmul

Matmul is just a bunch of dot products between the rows of one matrix and the columns of another: i.e. `c[i,j]` is the dot product of row `a[i, :]` and column `b[:, j]`.

Let's consider the case of 3x3 matrices. `a` is:

```python
tensor([[1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.]], dtype=torch.float64)
```

`b` is:

```python
tensor([[0., 1., 2.],
        [3., 4., 5.],
        [6., 7., 8.]], dtype=torch.float64)
```

Let's derive the code above looking purely through modifying the shape of `a`.

1. `a` has shape `(3,3)`
2. `a[0]`, first row of `a`, has shape `(3,)` and val `[1, 1, 1]`
3. `a[i, :, None]` (or `a[i].unsqueeze(-1)`) has shape `(3,1)` and val `[[1], [1], [1]]`

Now multiplying the result of 3 by the matrix `b` is represented by the expression (I have put brackets in to denote the array dimensions):


$$
\left(\begin{matrix}(1)\\(1)\\(1)\end{matrix}\right) \times \left(\begin{matrix}(0&1&2)\\(3&4&5)\\(6&7&8)\end{matrix}\right)
$$


From the rules of broadcasting, the $(1)$s on the left array are expanded to match the size of the rows on the right array (size 3). As such, the full expression computed effectively becomes:


$$
\left(\begin{matrix}(1&1&1)\\(1&1&1)\\(1&1&1)\end{matrix}\right)
\times
\left(\begin{matrix}(0&1&2)\\(3&4&5)\\(6&7&8)\end{matrix}\right)
=
\left(\begin{matrix}(0&1&2)\\(3&4&5)\\(6&7&8)\end{matrix}\right)
$$


The final step is to `sum(dim=0)`, which sums up all the rows leaving a vector of shape `(3,)`, value: `[ 9., 12., 15.]	`. That completes the dot product and forms the first row of matrix `c`. Simply repeat that for the remaining 2 rows of `a` and you get the final result of the matmul:

```python
tensor([[ 9., 12., 15.],
        [18., 24., 30.],
        [27., 36., 45.]], dtype=torch.float64)
```



### Einstein Summation Matmul

This will be familiar to anyone who studied Physics, like me! Einstein summation (`einsum`) is a compact representation for combining products and sums in a general way. From the numpy docs:

*"The subscripts string is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding operand.  Whenever a label is repeated it is summed, so `np.einsum('i,i', a, b)` is equivalent to `np.inner(a,b)`. If a label appears only once, it is not summed, so `np.einsum('i', a)` produces a view of a with no changes."*

```python
def matmul(a,b): 
    return torch.einsum('ik,kj->ij', a, b)
```

Time: __172µs__

_This is super concise with great performance, but also kind of gross. It's a bit  weird that `einsum` is a mini-language that we pass as a Python string. We get no linting or tab completion benefits that you would get if it were somehow a first class citizen in the language. I think `einsum` could certainly be great and quite readable in cases where you are doing summations on tensors with lots of dimensions._



### PyTorch Matmul Intrinsic

Matmul is already provided by PyTorch (or Numpy) using the `@` operator:

```python
def matmul(a, b):
	return a@b
```

Time: __31.2µs__

_The best performance is, unsuprisingly, provided by the library implementation. This operation will drop down to an ultra optimized library like BLAS or cuBLAS, written by low-level coding warrior-monks working at Intel or Nvidia who have have spent years hand optimizing linear algebra code in C and assembly. (The matrix multiply algorithm is actually a very complicated topic, and no one knows what the fastest possible algorithm for it is. See [this wikipedia page](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm) for more). So basically in the real world, you should probably avoid writing your own matmal!_



## Single Layer Network: Forward Pass

Work through the Jupyter notebook: [02_fully_connected.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02_fully_connected.ipynb) 

Create simple network for MNIST.  One hidden layer and one output layer, parameters:

```python
n = 50000 
m = 784
nout = 1 # just for teaching purposes here, should be 10
nh = 50
```

The model will look like this:


$$
X \rightarrow \mbox{Lin}(W_1, b_1) \rightarrow \mbox{ReLU} \rightarrow \mbox{Lin2}(W_2, b_2) \rightarrow \mbox{MSE} \rightarrow L
$$


**Linear** activation function:

```python
def lin(x, w, b):
	return x@w + b
```



 **ReLU** activation function:

```python
def relu(x):
	return x.clamp_min(0.)
```



**Loss function** we'll use here is the _Mean Squared Error (MSE)_. This doesn't quite fit for a classification task, but it's used as a pedgogical tool for teaching the concepts of loss and backpropagation.

```python
def mse(output, targ):
	return (output.squeeze(-1) - targ).pow(2).mean()
```



__Forward Pass__ of model:

```python
def model(xb):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3

preds = model(x_train)
loss = mse(preds, y_train)
```



Let's go over the tensor dimensions to review how the forward pass works:

- Input $X$ is a batch of vectors of size 784,  `shape=[N, 784]`
- Hidden layer is of size 50 and has an input of `shape=[N, 784]` =>  $W_1$: `shape=[784, 50]`,  $b_1$: `shape=[50]`, output: `shape=[N, 50]`
- Output layer has size 1 and input of `shape=[N, 50]` => $W_2$: `shape=[50, 1]`, $b_2$: `shape=[1]`, output: `shape=[N, 1]`

## Initialisation

Recent research shows that weight initialisation in NNs is actually super important. If the network isn't initialised well, then after one pass through the network the output can sometimes become vanishingly small or even explode, which doesn't bode well for when we do backpropagation.

A rule of thumb to prevent this is:

1. The _mean_ of the activations should be zero
2. The _variance_ of the activations should stay close to 1 across every layer.

Let's try Normal initialisation with a linear layer:

```python
w1 = torch.randn(m,nh)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)
b2 = torch.zeros(1)
```

```python
x_valid.mean(),x_valid.std()
>>> (tensor(-0.0059), tensor(0.9924))
```

```python
t = lin(x_valid, w1, b1)
t.mean(),t.std()
>>>	(tensor(-1.7731), tensor(27.4169))
```

After one layer, it's already in the rough.

A better initialisation is Kaiming/He initialisation ([paper](https://arxiv.org/abs/1502.01852)). For a linear activation you simply divide by the square root of the number of inputs to the layer.:

```python
w1 = torch.randn(m,nh)/math.sqrt(m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)/math.sqrt(nh)
b2 = torch.zeros(1)
```

 Test:

```python
t = lin(x_valid, w1, b1)
t.mean(),t.std()
>>> (tensor(-0.0589), tensor(1.0277))
```

The initialisation used __depends on the activation function used__. If we instead use a ReLU layer then we have to do something different from the linear.

```python
def relu(x):
	return x.clamp_min(0.)

t = relu(lin(x_valid, w1, b1))
```

If you have a normal distribution with mean 0 with std 1, but then clamp it at 0, then obviously the resulting distribution will no longer have mean 0 and std 1.

> From pytorch docs: `a: the negative slope of the rectifier used after this layer (0 for ReLU by default)`
>
> $$\text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}$$
>
> This was introduced in the paper that described the Imagenet-winning approach from *He et al*: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), which was also the first paper that claimed "super-human performance" on Imagenet (and, most importantly, it introduced resnets!)

```python
w1 = torch.randn(m,nh)*math.sqrt(2/m)
```

Test:

```python
t = relu(lin(x_valid, w1, b1))
t.mean(),t.std()
>>> (tensor(0.5854), tensor(0.8706))
```

The function that does this in the Pytorch library is:

```python
from torch.nn import init

w1 = torch.zeros(m,nh)
init.kaiming_normal_(w1, mode='fan_out')
```

`'fan_out'` means that we divide by `m`, while `'fan_in'` would mean we divide by `nh`. Here it helps to think of 'out' and 'in' here from the perspective of back-propagation, running backwards through the network. 

Let's get a better view of the means and standard deviations of the model with Kaiming initialization by running it a few thousand times and looking at the distributions:

![img](/images/fastai/Sat, 01 Feb 2020 222711.png)

![img](/images/fastai/Sat, 01 Feb 2020 222749.png)

The means have a clearly Gaussian distribution with mean value 0.01. The standard deviations have a slightly skewed distribution, but the mean value is 1.17.  We see empirically that the expected output values of the model after Kaiming initialisation are approximately mean 0, standard deviation 1.



### Aside: Init in Pytorch - sqrt(5)??

In `torch.nn.modules.conv._ConvNd.reset_parameters`:

```python
def reset_parameters(self):
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
```

A few differences here:

1. Uses Uniform distribution instead of a Normal distribution. This just seems to be convention the Pytorch authors have chosen to use. Not an issue and it is centred around zero anyway.
2. The `sqrt(5)` is probably a bug, according to Jeremy.

The initialization for the linear layer is similar.

From the documentation on parameter `a`:

> a: the negative slope of the rectifier used after this layer (0 for ReLU
>             by default)

For ReLU it should be 0, but here it is hard-coded to `sqrt(5)`. So for ReLu activations in Conv layers, the initialization of some layers in Pytorch is suboptimal by default. 

TODO: link to the issue page in PyTorch. Has this been resolved in the main branch yet?



## Gradients and Backpropagation

To understand backpropagation we need to first understand the chain rule from calculus. The model looks like this:


$$
x \rightarrow \mbox{Lin1} \rightarrow \mbox{ReLU} \rightarrow \mbox{Lin2} \rightarrow \mbox{MSE} \rightarrow L
$$



Where $L$ denotes the loss. We can also write this as:


$$
L = \mbox{MSE}(\mbox{Lin2}(\mbox{ReLU}(\mbox{Lin1(x)})), y)
$$

Or fully expanded:


$$
\begin{align}
L &= \frac{1}{N}\sum_n^N\left((\mbox{max}(0, X_nW^{(1)} + b^{(1)})W^{(2)} + b^{(2)}) - y_n\right)^2
\end{align}
$$

In order to update the parameters of the model, we need to know what is the gradient of $L$ with respect to (wrt) the parameters of the model. What are the parameters of this model? They are: $W^{(1)}_{ij}$,  $W^{(2)}_{ij}$,   $b^{(1)}_i$, $b^{(2)}_i$ (including indices to remind you of the tensor rank of the parameters). The partial derivatives of the parameters we want to calculate are:


$$
\frac{\partial L}{\partial W^{(1)}_{ij}}, \frac{\partial L}{\partial W^{(2)}_{ij}}, \frac{\partial L}{\partial b^{(1)}_{i}}, \;\mbox{and}\; \frac{\partial L}{\partial b^{(2)}_{i}}
$$


On first sight, looking at the highly nested function of $L$ finding the derivative of it wrt to matrices and vectors looks like a brutal task. However the cognitive burden is greatly decreased thanks to [_the chain rule_](https://en.wikipedia.org/wiki/Chain_rule). 

When you have a nested function, such as:


$$
f(x,y,z) = q(x, y)z \\
q(x,y) = x+y
$$

The chain rule tells you that the derivative of $f$ wrt to $x$ is:


$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial x} = (z)\cdot(1) = z
$$


A helpful mnemonic is to picture the $\partial q$'s 'cancelling out'. 



### Backpropagation: Graph Model

How does this fit into backpropagation? Things become clearer when the model is represented as a computational graph, instead of as equations. 

Imagine some neuron $f$ in the middle of a bigger network. In the forward pass, data $x$ and $y$ flows from left to right through the neuron $f$, outputting $z$,  then calculating the loss $L$. Then we want the gradients of all the variables wrt the loss. Here is a diagram taken from [CS231 course](http://cs231n.stanford.edu/) :

![image-20200205212902584](/images/fastai/image-20200205212902584.png)

*([Source](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture04.pdf): brilliant CS231 course from Stanford. This lecture made backpropagation 'click' for me: [video](https://youtu.be/GZTvxoSHZIo), [notes](http://cs231n.github.io/optimization-2/)).*

Calculate the gradients of the variables backwards from right to left. We have the gradient $\frac{\partial L}{\partial z}$ coming from 'upstream'.  To calculate $\frac{\partial L}{\partial x}$, we use the chain rule:


$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}
$$


The **gradient = upstream gradient $\times$ local gradient**. This relation recurses back through the rest of the network, so a neuron directly before $x$ would receive the upstream gradient $\frac{\partial L}{\partial x}$. The beauty of the chain rule is that it enables us to break up the model into its constituent operations/layers, compute their local gradients, then multiply by the gradient coming from upstream, then *propagate the gradient backwards*, repeating the process.

Coming back to our model - $\mbox{MSE}(\mbox{Lin2}(\mbox{ReLU}(\mbox{Lin1(x)})), y)$ - to compute the backward pass we just need to compute the expressions for the derivatives of MSE, Linear layer, and ReLU layer.

### Gradients of Vectors or Matrices

What happens when $z$, $x$, and $y$ aren't scalar, but are vectors or matrices? Nothing changes with how backpropagation works - just the maths for computing the local gradients gets a bit hairier. 

If the loss $L$ is a scalar and $\mathbf{z}$ is a vector then the derivative would be __vector__:


$$
\frac{\partial L}{\partial \mathbf{z}} = \left(\frac{\partial L}{\partial z_1}, \frac{\partial L}{\partial z_2}, ...,\frac{\partial L}{\partial z_n}, \right)
$$


Think: *"For each element of $\mathbf{z}$, if it changes by a small amount how much will $L$ change?"*

If $\mathbf{x}$ and $\mathbf{z}$ are both vectors then the derivative would be a __Jacobian matrix__:


$$
\mathbf{\frac{\partial \mathbf{z}}{\partial \mathbf{x}}} = 
\left[\begin{array}{ccc}
\frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} & ... & \frac{\partial z_1}{\partial x_m} \\
    \frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2} & ... & \frac{\partial z_2}{\partial x_m} \\
    ... & ... & ... & ...\\
    \frac{\partial z_n}{\partial x_1} & \frac{\partial z_n}{\partial x_2} & ... & \frac{\partial z_n}{\partial x_m}
\end{array}\right]
$$


Think: *"For each element of $\mathbf{x}$", if it changes by a small amount then how much will each element of $\mathbf{y}$ change?*

Summary, again taken from [CS231n](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture04.pdf):

![image-20200206010837769](/images/fastai/image-20200206010837769.png)

**More info**: a full tutorial on matrix calculus is provided here: [Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/).



### Gradient of MSE

The mean squared error:


$$
L = \frac{1}{N} \sum_i^N (z_i - y_i)^2
$$

Where $N$ is the batch size, $z_i$ is the output of the model for data point $i$, and $y_i$ is the target value of $i$. The loss is the average of the squared error in a batch. $\mathbf{z}$ is a vector here. The derivative of scalar $L$ wrt a vector will be vector. 


$$
\frac{\partial L}{\partial z_i} = \frac{\partial}{\partial z_i}\left(\frac{1}{N}\sum_j^N (z_j - y_j)^2\right) = \frac{\partial}{\partial z_i} \frac{1}{N} (z_i - y_i)^2 = \frac{2}{N}(z_i - y_i)z_i
$$


All the other terms in the sum go to zero because they don't depend on $z_i$. Notice also how $L$ doesn't appear in the gradient - we don't actually need the value of the loss in the backwards step!

In Python code:

```python
def mse_grad(inp, targ):
    # inp from last layer of model, shape=(N,1)
    # targ targets, shape=(N)
    # want: grad of MSE wrt inp, shape=(N, 1)
    grad = 2. * (inp.squeeze(-1) - targ).unsqueeze(-1) / inp.shape[0]
    inp.g = grad
```



### Gradient of Linear Layer

Linear layer:


$$
Y = XW + b
$$

Need to know:


$$
\frac{\partial L}{\partial X}, \frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}
$$


Where $X$, and $W$ are matrices and $b$ is a vector. We already know $\frac{\partial L}{\partial Y}$ - it's the _upstream gradient_ (remember it's a tensor, not necessarily a single number).

Here is where the maths gets a bit hairier. It's not worth redoing the derivations of the gradients here, which can be found in these two sources: [matrix calculus for deep learning](https://explained.ai/matrix-calculus),  [linear backpropagation](http://cs231n.stanford.edu/handouts/linear-backprop.pdf).

The results:


$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}W^T \\
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y} \\
\frac{\partial L}{\partial b_i} = \sum_j^M \frac{\partial L}{\partial y_{ij}}
$$


In Python:

```python
def lin_grad(inp, out, w, b):
    # inp - incoming data (x)
    # out - upstream data 
    # w - weight matrix
    # b - bias
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(dim=0)
```



### Gradient of ReLU

Gradient of ReLU is easy. For the local gradient - if the input is less than 0, gradient is 0, otherwise it's 1. In Python

```python
def relu_grad(inp, out):
	# inp - input (x)
    # out - upstream data
    inp.g = (inp>0).float() * out.g
```



### Putting it together: forwards and backwards

```python
def forwards_and_backwards(inp, targ):
	# forward pass
    l1 = lin(inp, w1, b1)
    l2 = relu(l1)
    out = lin(l2, w2, b2)
    loss = mse(out, targ)
    
    # backward pass
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```



### Refactoring

The rest of the notebook - [02_fully_connected.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02_fully_connected.ipynb) - is spent refactoring this code using classes so we understand how pytorch's classes are constructed. I won't reproduce it here. The end result with pytorch's classes is:

```python
from torch import nn


class Model(nn.Module):
	def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in, nh), nn.ReLU(), nn.Linear(nh, n_out)]
        self.loss = mse
    def __call__(self, x, targ):
        for l in self.layers:
            x = l(x)
        return self.loss(x.squeeze(), targ)
```



Now we understand how backprop works, we luckily don't have to derive anymore derivatives of tensors, we can instead from now on harness pytorch's autograd to do all the work for us!

```python
model = Model(m, nh, 1)
loss = model(x_train, y_train)
loss.backward() # do the backward pass!
```



## Links and References

- Lesson 8 [lecture video](https://course.fast.ai/videos/?lesson=8).

- [Lesson notes from Laniken](https://medium.com/@lankinen/fast-ai-lesson-8-notes-part-2-v3-8965a6532f51) provide a transcription of the lesson.

- Broadcasting tutorial from Jake Vanderplas: [Computation on Arrays: Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).

- Deeplearning.ai notes on initialisation with nice demos of different initialisations and their effects: [deeplearning.ai](https://www.deeplearning.ai/ai-notes/initialization/)

- Kaiming He paper on initialization with ReLu activations (**assignment**: read section 2.2 of this paper): [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

- Fixup Initialization: [paper](https://arxiv.org/abs/1901.09321) where they trained a 10,000 layer NN with no normalization layers through careful initialization.

- Things that made Backprop 'click' for me:

  - CS231: backpropagation explained using the a circuit model: [http://cs231n.github.io/optimization-2/](http://cs231n.github.io/optimization-2/)
  - [CS231: backpropagation lecture (Andrej Karpathy)](https://youtu.be/i94OvYb6noo), [slides](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture04.pdf).
  - [Blog post](https://amva4newphysics.wordpress.com/2017/03/28/understanding-neural-networks-part-ii-back-propagation/) with worked examples of backpropagation on simple calculations.
  - [Calculus on Computational Graphs, Chris Olah.](https://colah.github.io/posts/2015-08-Backprop/)

  
