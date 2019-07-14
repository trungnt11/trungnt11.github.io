---
layout: post
title: Fast.ai Lesson 2 Notes
date: 2019-07-12
tags: deep-learning machine-learning fastai
description: My personal notes on Lesson 2 of part 1 of fast.ai (2019) -- <b>Data cleaning and production; SGD from scratch</b>. 
featured_image: fastai/image-20190706182251357.png
---




![image-20190706182251357](/images/fastai/image-20190706182251357.png)

_[Link to Lesson 2 lecture](https://course.fast.ai/videos/?lesson=2)_

## Overview of Lesson

This lesson has two parts. The first part is about constructing a image classifier from your own data. It details data collection from Google images, creating a validation set, and cleaning the data using the model. 

In the second part, we construct a simple linear model from scratch in PyTorch and train it using _gradient descent_ and _stochastic gradient descent_. That part got quite lengthy so I made it its own blog post [here]({% post_url 2019-07-13-sgd-from-scratch-fast-ai %}).



## Download Your Own Image Data

There is a trick to downloading data from google images. You can do the search manually for the images, then run some javascript magic to get the URLs for the images. You can then save these in a file and then download them from the command line.

1. Go to Google images and search for your desired images.

2. Open the browser javascript console: (⌘+⎇+J on Mac, Crtl+Shift+J on Windows/Linux).

3. Run the following the console:

   ```javascript
   urls = Array.from(document.querySelectorAll('.rg_di.rg_meta')).map(el=>JSON.parse(el.textContent).ou);
   window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
   ```

4. This initiates a download of a CSV that contains all the urls to the images shown on Google images.

5. Use fastai's `download_images` function and pass it the path to the CSV file as the argument.

6. Remove images that aren't valid. Use fastai's `verify_images` to delete these.



## Then Train With A CNN

Following the steps from Lesson 1:

1. Load data using the DataBunch API:

   ```python
   np.random.seed(42) # fix seed for to get same validation set
   data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 
                                     ds_tfms=get_transforms(), size=224, 
                                     num_workers=4).normalize(imagenet_stats)
   ```

2. Create the CNN learner and specify the architecture:

   ```python
   learn = create_cnn(data, models.resnet34, metrics=error_rate)
   ```

3. First fit the _head_ of the pretrained CNN with a few cycles:

   ```python
   learn.fit_one_cycle(4)
   ...
   ```

4. Then _unfreeze_ the _body_ of the pretrained CNN:

   ```python
   learn.unfreeze()
   ```

5. Run the learning rate finder:

   ```python
   learn.lr_find()
   ```

6. Inspect the learning rate graph and find the strongest downward slope whose negative trend persists for while with the increasing learning rate. Try to pick a learning rate corresponding to the steepest part of this slope.

   ![img](https://github.com/hiromis/notes/raw/master/lesson2/13.png)

7. Train the whole network again for a few cycles using a range of learning rates for each layer group, with the learning rate you picked being the highest. This is called __Discriminative Layer Training__ in fastai.

   ```python
   learn.fit_one_cycle(2, max_lr=slice(3e-5, 3))
   ```

In the Bear example Jeremy does this produces an error rate of 1.4% with a few hundred images and a few minutes of training time on a GPU.



## Intepretation

For a classification task such as the Bear example in the lecture, you want to look at the confusion matrix to see where it is failing (well, except where you have loads of classes). FastAI has a handy class for interpreting classification results:

```python
interp = ClassificationInterpretation.from_learner(learn)
```

```python
interp.plot_confusion_matrix()
```

![img](https://github.com/hiromis/notes/raw/master/lesson2/14.png)

Pretty good - only one mistake!



## Cleaning Up Your Dataset

Maybe there is noise or mistakes in your dataset. If we download images from google then perhaps there are images that are of the wrong thing. We want to clean it up.
Here is where human intelligence and a computer learner can be combined! It's very unlikely that a mislabeled data is going to be predicted correctly and with high confidence. We can look at the instances that the computer learner is least confident about - i.e. the instances with the highest loss. There is a nice widget for Jupyter notebook for inspecting and deleting things called `FileDeleter`:

```python
from fastai.widgets import *

losses,idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]
```

After cycling through `FileDeleter` and deleting the bad data you should eventually see fewer and fewer bad data points. At this point you are done and should retrain your model on the cleaned dataset.

Generally these CNN models are pretty good at handling small amounts of noise so this data cleaning will normally give you a small improvement.



## Putting your Model into Production

You probably want to use CPU for inference, except for massive scale (and you almost certainly don't need to train in real time). GPU is only effective if you can get things into neat batches with sizes like 64, which exploits the GPU parallelism. In PyTorch you can specify CPU via:

```python
fastai.defaults.device = torch.device('cpu')
```



Let's use the trained model for _inference_. We upload an image of a bear and store that in a variable `img`:

```python
img = open_image(path/'black'/'00000021.jpg')
```

![img](https://github.com/hiromis/notes/raw/master/lesson2/bear.png)



And as per usual, we created a data bunch, but this time, we're not going to create a data bunch from a folder full of images, we're going to create a special kind of data bunch which is one that's going to grab one single image at a time. So we're not actually passing it any data. The only reason we pass it a path is so that it knows where to load our model from. That's just the path that's the folder that the model is going to be in.

You also need to pass it the same transformations , size, and normalizations that you used when training the CNN. You then `create_cnn` with this fake dataset and then load the weights that were saved in the training phase:

```python
classes = ['black', 'grizzly', 'teddys']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), 																											 size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load('stage-2')
```

Then prediction is done using the `predict` method and passing in the real single image data:

```python
pred_class,pred_idx,outputs = learn.predict(img)
pred_class

> 'black'
```



This is the engine of an web-app. The rest of the app can be coded up in a framework like Flask or Starlette. Here is a great example that uses Starlette: [cougar-or-not](https://github.com/simonw/cougar-or-not).

There are services for hosting, such as: https://www.pythonanywhere.com/



## Things That Can Go Wrong

The problems will basically be either:

- The learning rate is too high or too low
- The number of epochs is too many or too few

__Learning rate too high__: basically ruins everything and results in a super high validation loss

__Learning rate too low__: the error rate goes down _really slowly_. The other thing you see if your learning rate is too small is that your training loss will be higher than your validation loss. You never want a model where your training loss is higher than your validation loss. That always means you are _under-fitting_ which means either your learning rate is too low or your number of epochs is too low. So if you have a model like that, train it some more or train it with a higher learning rate.

__Number of epochs too few__: training loss much higher than validation loss, which is a symptom of _under-fitting_. It needs to learn more.

__Number of epochs too many__: Too many epochs create something called "overfitting". If you train for too long as we're going to learn about it, it will learn to recognize your particular teddy bears but not teddy bears in general.



### The Truth About Overfitting

The only thing that tells you you are overfitting is that the error rate improves for a while and then starts getting worse again.

__Myth__: If the training loss is less than the validation loss then the model is overfitting. __Absolutely not true__.

> Any model that is trained correctly will always have a lower training loss than validation loss

You want your model to have a low error. So as long as you're training and your model error is improving, you're not overfitting.

In Jeremy's option, despite what you hear, it's actually very hard to overtrain in deep learning.



### Underfitting

How can the training loss be _higher_ than the validation loss? This doesn't really seem like it could happen except if you had some contrived validation set. It can however happen quite easily with training neural networks because of __dropout__. 

Dropout is _turned on_ while training and _turned off_ in the validation. If the result is made much worse by dropout then it means that the network has not learned sufficiently well and it is therefore underfitting. Ways to fix this are: train with more epochs, use higher learning rate, use less dropout, or adjust weight decay parameters. 



## SGD From Scratch

This part kind of outgrew this blog post so I have spun this out into its own blog post [here]({% post_url 2019-07-13-sgd-from-scratch-fast-ai %}).



## Jeremy Says...

1. If forum posts are overwhelming, click “summarize this topic” at the bottom of the first post. (Only works for >50 replies).
2. Please follow the official server install/setup instructions, they work and are easy.
3. It’s okay to feel intimidated, there’s a lot, but just pick one piece and dig into it. Try to push a piece of code, or learn a concept like regular expressions, or create a classifier, or whatever. Context: [Lesson 2: It’s okay to feel intimidated 30](https://youtu.be/ccMHJeQU4Qw?t=600)
4. If you’re stuck, keep going. See image below! Context: [Lesson 2: If you’re stuck, keep going 38](https://youtu.be/ccMHJeQU4Qw?t=867)
5. If you’re not sure which learning rate is best from plot, try both and see.
6. When you put a model into production, you probably want to use CPU for inference, except at massive scale. Context: [Lesson 2: Putting Model into Production 17](https://youtu.be/ccMHJeQU4Qw?t=2308)
7. Most organizations spend too much time gathering data. Get a small amount first, see how it goes.
8. If you think you’re not a math person, check out Rachel’s talk: [There’s no such thing as “not a math person” 56](https://youtu.be/q6DGVGJ1WP4). My own input: only 6 minutes, everyone should watch it!

![keepgoing](https://forums.fast.ai/uploads/default/optimized/3X/d/e/de73a146088bb62668b7e2d0215b398d9452177e_2_690x422.png)



## Q & A

- _When generating new image dataset, how do you know how many images are enough? What are ways to measure “enough”?_

  > That’s a great question. Another possible problem you have is you don’t have enough data. How do you know if you don’t have enough data? Because you found a good learning rate (i.e. if you make it higher than it goes off into massive losses; if you make it lower, it goes really slowly) and then you train for such a long time that your error starts getting worse. So you know that you trained for long enough. And you’re still not happy with the accuracy﹣it’s not good enough for the teddy bear cuddling level of safety you want. So if that happens, there’s a number of things you can do and we’ll learn pretty much all of them during this course but one of the easiest one is get more data. If you get more data, then you can train for longer, get higher accuracy, lower error rate, without overfitting.
  >
  > Unfortunately, there is no shortcut. I wish there was. I wish there’s some way to know ahead of time how much data you need. But I will say this﹣most of the time, you need less data than you think. So organizations very commonly spend too much time gathering data, getting more data than it turned out they actually needed. So get a small amount first and see how you go.

- _What do you do if you have unbalanced classes such as 200 grizzly and 50 teddies?_

  > Nothing. Try it. It works. A lot of people ask this question about how do I deal with unbalanced data. I’ve done lots of analysis with unbalanced data over the last couple of years and I just can’t make it not work. It always works. There’s actually a paper that said if you want to get it slightly better then the best thing to do is to take that uncommon class and just make a few copies of it. That’s called “oversampling” but I haven’t found a situation in practice where I needed to do that. I’ve found it always just works fine, for me.

  

## Links and References

- Parts of my notes have been copied from the excellent lecture transcriptions made by @hiromi. Link: [Lesson2 Detailed Notes](https://github.com/hiromis/notes/blob/master/Lesson2.md).
- This is an in-depth tutorial on PyTorch: <https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e>

- [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/) by @rachel

- [There's no such thing as "not a math person"](https://www.youtube.com/watch?v=q6DGVGJ1WP4) by @rachel

- [Responder](https://github.com/kennethreitz/responder) - a web app framework built on top of Starlette

- Post about an [alternative image downloader/cleaner](https://www.christianwerner.net/tech/Build-your-image-dataset-faster/) by @cwerner

- [A tool for excluding irrelevant images from Google Image Search results](https://forums.fast.ai/t/tool-for-deleting-files-on-the-google-image-search-page-before-downloading/28900) by @melonkernel

- [Machine Learning is Fun](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721) - source of image/number GIF animation shown in lesson

- [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381), mentioned by Jeremy as a way to solve imbalanced datasets.