---
layout: post
title: Fast.ai Lesson 1 Notes
date: 2019-07-06
tags: deep-learning machine-learning
description: My personal notes on Lesson 1 of part 1 of fast.ai (2019) -- <b>Image classification</b>. 
featured_image: fastai/image-20190706182251357.png
---




![image-20190706182251357](/images/fastai/image-20190706182251357.png)

##  Task 1 - World Class Image Classifier

- Task 1: image classification using transfer learning.
- Cat breeds and dog breeds dataset from Oxford: [Cats and Dogs Breeds Classification Oxford Dataset \| Kaggle](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset).
- Previous fastai courses used the Cats vs Dogs dataset, but it's too easy to get SOTA performance with this dataset using transfer learning these days!

##  Load the Data

- FastAI has its own way of handling different datasets: the _DataBunch API_.

- This integrates the loading of different types of data, the labeling, splitting into train/val/test, and the data transformations for standardisation, normalisation, and training augmentation.

- To load the cats and dogs dataset:
  
  ```python
  pat = r'/([^/]+)_\d+.jpg$'
  data = ImageDataBunch.from_name_re(path_img, 
                                   fnames, 
                                   pat, 
                                   ds_tfms=get_transforms(),
                                   size=224,
                                   bs=64,
                                   ).normalize(imagenet_stats)
  ```

- What is this doing? Let's look at the docs:

  ```python
  class ImageDataBunch(fastai.basic_data.DataBunch)
   |  DataBunch suitable for computer vision.
   ...
   |  from_name_re(path:Union[pathlib.Path, str], 
                   fnames:Collection[pathlib.Path], 
                   pat:str, 
                   valid_pct:float=0.2, 
                   **kwargs) from builtins.type
   |      Create from list of `fnames` in `path` with re expression `pat`.
  ```

- `get_transforms` is a function that returns a list of default image transformations for data augmentation.
- `size=224` resizes the images to `224x224`. This is the size that the network we are using (resnet34) has been trained on.
- `bs` is the batchsize.
- `normalize(imagenet_stats)` normalizes the images so that the pixel values are between 0 and 1 (necessary for using the neural network). The network has been pretrained on imagenet data so we need to normalize our new data with respect to the imagenet data. This way the images are placed near the distribution that the network was trained on and so gives the network data that it is 'used to' seeing.

  


##  Training a Model using a Pretrained ResNet

- ResNet generally 'just works well' for image tasks.
- Here's how to create a CNN with the fastai library:

  ```python
  learn = create_cnn(data, models.resnet34, metrics=error_rate)
  ```

- `data` is the DataBunch object of the cats/dogs data we created earlier.

- Here we are using a variant of resnet called `resnet34`. The 34 simply means it has 34 layers. There are others avaiable with 18, 50, and more layers. 

- __One-cycle policy__:
  
  - `learn.fit_one_cycle(4)`
  - People train neural networks using Stochastic Gradient Descent (SGD). Here the training set is divided into random batches (say of size 64) and the network weights are updated after each batch. After the network has seen all the items in the training set, this is called an _epoch_. The rate at which the weights are changed is called the __learning rate__. Typically people set this to a single value that remains unchanged during an epoch.
  - One cycle policy is a way of training the neural network using SGD faster by varying the learning rate during each epoch.
  - Sylvain explains:
    
    > He recommends to do a cycle with two steps of equal lengths, one going from a lower learning rate to a higher one than go back to the minimum. The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower. Then, the length of this cycle should be slightly less than the total number of epochs, and, in the last part of training, we should allow the learning rate to decrease more than the minimum, by several orders of magnitude.
  - The momentum also varies correspondingly:
    
    ![ef1822a7](/images/fastai/ef1822a7.png)
  - When the learning rate is high we want momentum to be lower. This enables the SGD to quickly change directions and find a flatter parameter space.
  - Link: [Another data science student's blog – The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)

- __Learning Rate Finder__
  
  - The method is basically successively increasing $\eta$ every batch using either a linear or exponential schedule and looking the loss. While $\eta$ has a good value, the loss will be decreasing. When $\eta$ gets too large the loss will start to increase. You can plot the loss versus $\eta$ and see by eye a learning rate that is largest where the loss is decreasing fastest.



## Jeremy Says...

1. Don’t try to stop and understand everything.
2. Don’t waste your time, learn Jupyter keyboard shortcuts. Learn 4 to 5 each day.
3. Please run the code, really run the code. Don’t go deep on theory. Play with the code, see what goes in and what comes out.
4. Pick one project. Do it really well. Make it fantastic.
5. Run this notebook (lesson1-pets.ipynb), but then get your own dataset and run it! (extra emphasis: do this!) If you have a lot of categories, don’t run confusion matrix, run… `interp.most_confused(min_val=n)`



## Q & A

- _As GPU mem will be in power of 2, doesn’t size 256 sound more practical considering GPU utilization compared to 224?_

  > The brief answer is that the models are designed so that the final layer is of size 7 by 7, so we actually want something where if you go 7 times 2 a bunch of times (224 = 7x2x2x2x2x2), then you end up with something that’s a good size. 
  > Objects often appear in the middle of an image in the ImageNet dataset. After 5 maxpools, a 224x224 will be 7x7 meaning that it will have a centerpoint. A 256x256 image will be 8x8 and not have a distinct centerpoint.

- _Why resnet and not inception architecture?_

  > Resnet is Good Enough!
  > See the DAWN benchmarks - the top 4 are all Resnet.You can consider different models for different use cases.
  > For example, if you want to do edge computing, mobile apps, Jeremy still suggests running the model on the local server and port results to the mobile device. But if you want to run something on the low powered device, there are special architectures for that.
  >
  > Inception is pretty memory intensive.
  > fastai wants to show you ways to run your model without much fine-tuning and still achieve good results.
  > The kind of stuff that always tends to work. Resnet works well on a lot of image classification applications.

- _Will the library use multi GPUs in parallel by default?_

  > The library will use multiple CPUs by default but just one GPU by default. We probably won’t be looking at multi GPU until part 2. It’s easy to do and you’ll find it on the forum, but most people won’t be needing to use that now.



## Links

- [Stanford DAWN Deep Learning Benchmark (DAWNBench) ·](https://dawn.cs.stanford.edu/benchmark/)
- [[1311.2901] Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- [Another data science student's blog – The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)
