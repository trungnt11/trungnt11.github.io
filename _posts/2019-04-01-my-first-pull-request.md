---
layout: post
title: My First Pull Request
date: 2019-04-01
tags: python fastai
description: My first pull request to an open source probject was merged into fastai.
comments: true
---


I'm very proud to have made my first open source contribution! :-) I added a feature to the [FastAI](https://github.com/fastai/fastai) deep learning library to make its data types 'countable' and thus work with the `collections.Counter` class.



## Problem 

While I was working through the image classification homework from [lesson 2](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb). I wanted to check the how many images of each class there were in my data block. The best way to count the number of values in a collection in python is to use the `collections.Counter` class which creates a dictionary mapping value to count.

However when I tried this with the data block I got this:

``` python
> Counter(data.train_ds.y)

  Counter({
    Category chimp        1
    Category gorilla      1
    Category gorilla      1
    Category chimp        1
    Category gorilla      1
    Category gorilla      1
    Category gorilla      1
    Category gorilla      1
    Category gorilla      1
      ...

```



## Solution

This problem is caused by fact that there was no `__eq__` implemented for the `Category` class. When different `Category` objects were compared python's default equality would only check whether they were literally the same object rather than checking their values. To get an object to work with a dictionary class in python you also have to implement a `__hash__` method.

I confirmed this with the hot patch:

```python
> Category.__eq__ = lambda self, that: self.data == that.data
> Category.__hash__ = lambda self: hash(self.obj)
> Counter(data.train_ds.y)

	Counter({Category orangutan: 56, Category gorilla: 177, Category chimp: 173})
```

With Sylvain Gugger's guidance, I then implemented `__eq__` method properly in fastai for the ground class `ItemBase` so that all of the different data classes in fastai could have equality. Hash didn't make sense for all the subclasses (like floats or arrays of numbers), so we compromised on implementing hash methods only on the subclasses where it made sense.

Here is the link to my pull request: [https://github.com/fastai/fastai/pull/1717](https://github.com/fastai/fastai/pull/1717).



## Aside: Making Python Objects Counter-Ready

In order to make your python objects play nice with dictionary's they need to override two python built-ins:

1. `__eq__`
2. `__hash__`

Suppose that the python object contains some value `val` that defines the object's uniquness. 

Let's create a python class called `Category`:

```python
class Cat: 
    def __init__(self, val): 
      	self.val = val 
    def __str__(self): 
      	return f'Cat({self.val})' 
    def __repr__(self): 
    		return f'Cat({self.val})' 
```



This class won't work properly with Counters:

```python
> xs = [Cat(2), Cat(2), Cat(1), Cat(3)]
> Counter(xs)
	Counter({Cat(2): 1, Cat(2): 1, Cat(1): 1, Cat(3): 1})

```



Equality doesn't work:

```python
> Cat(2) == Cat(2)
	False
```



Two objects with the same value don't have the same hash:

```python
> hash(Cat(2))
	-9223372036573193412
> hash(Cat(2))
	281542562
```



You have to implement the hash and equality built-ins:

```python
class Cat: 
    def __init__(self, val): 
      	self.val = val 
    def __str__(self): 
      	return f'Cat({self.val})' 
    def __repr__(self): 
    		return f'Cat({self.val})' 
    def __eq__(self, other):
    		return self.val == other.val
    def __hash__(self):
      	return hash(self.val)
```

Now it works:

```python
> xs = [Cat(2), Cat(2), Cat(1), Cat(3)] 
	Counter(xs)
	Counter({Cat(2): 2, Cat(1): 1, Cat(3): 1})
```



How does this work? The `Cat` objects are being used as keys in the Counter. When a new `Cat` object comes into the Counter we need to compare it with all the other keys already in the Counter. If a `Cat` object of the same value is there already then we need to increment the value associated with that `Cat` object. 

For efficiency, however, dictionaries in python don't store the keys in a big list rather in _buckets_. When a new `Cat` object comes into the Counter it is assigned to a bucket using its `hash` value. Here three things can happen.

1. If the bucket is empty then store the value there.
2. If the bucket isn't empty compare the incoming object with the objects there using `eq`. If they are the same, increment the counter. 
3. If they are different you have a __hash collision__. Store the incoming object in the bucket and set the counter value to 1.



**In summary**:

For correctness with dictionaries:

1. `obj1 == obj2` if `obj1.val == obj2.val`
2. `hash(obj1) == hash(obj2)` if `obj1.val == obj2.val`

If the object is to be used as a key in a Counter we need to be able to correctly compare it to other keys in the Counter. If two objects are equal then we know that they are the same key and we can increment the counter. Two objects with the same value need to be hashed to the same bucket.

For efficiency with dictionaries:

1. `hash(obj1) should ideally != hash(obj2)` if `obj1.val != obj2.val`

It is possible, though undesirable, that two objects with different values get hashed to the same bucket. This is called a __hash collision__. This isn't a correctness problem, rather an _efficiency problem_. Every hash collision is like an if statement in the dictionary to specially handle those cases. Ideally, every unique value should have its own unique hash so that there are no hash collisions.



## Links

- Link to pull request: [https://github.com/fastai/fastai/pull/1717](https://github.com/fastai/fastai/pull/1717)
- Link to forum post: [https://forums.fast.ai/t/get-value-counts-from-a-imagedatabunch/38784/21](https://forums.fast.ai/t/get-value-counts-from-a-imagedatabunch/38784/21)

