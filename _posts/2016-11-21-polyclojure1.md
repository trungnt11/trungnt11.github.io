---
layout: post
title: "Polymorphism in Clojure: A Tutorial Using Numbers, Part 1"
date: 2016-11-21
comments: false
tags: clojure polymorphism functional-programming
---

_This post is composed of 3 parts: Part 1, [Part 2]({% post_url 2016-11-21-polyclojure2 %}), [Part 3]({% post_url 2016-11-21-polyclojure3 %})_

## Introduction

For over a year in my spare time I have been working through the vaunted Structure and Interpretation of Computer Programs (SICP) using both Clojure and Racket. The book has a reputation for 'A-Ha!' moments, and I have experienced many of these while reading it. One of my favourite such moments occured in the final section of Chapter 2 on 'Data-directed programming'.

The concern of this section is in implementing a system for calculating with different types of numbers (e.g. complex, integers, rationals, etc). Variants of this example are commonly used  beginners books to Java and C++ in order to demonstrated how Object Oriented programming (OOP) can model data and abstractions. However SICP is not a language manual and doesn't concern itself with showing how to make some feature of a language do what you want. It instead focuses on the problem of trying to model numbers itself, and then creates the necessary structures to solve this problem. SICP uses this problem to motivate the purpose of OOP and it then creates a basic implementation of Objects in scheme. The book then goes beyond OOP and provides an implementation of multiple dispatch in scheme.

I found this to be such an enlightening exercise that I started thinking about how well other languages could solve this problem. In this blog post I want to implement the Numbers program in Clojure, a language that provides multiple dispatch. I hope that will be  a fairly rigorous tutorial for polymorphism in Clojure that demonstrates every feature for polymorphism in Clojure's core libraries.

## Complex Numbers
A complex number, $z$ is a number expressed in the form $z=u+iv$, where $u$ and $v$ are real numbers and $i^2=-1$. $u$ is called the _real part_ and $v$ is called the _imaginary part_. 

* We can represent a complex number as a pair $(u, v)$, called the _rectangular form_.

* An alternative representation is the _polar form_ where the complex number is represented by the pair $(r, \phi)$, where $r$ is the _magntitude_ and $\phi$ is the _angle_. 

Rectangular and polar forms are related via the following formulae:

$$\begin{eqnarray}
u &=& r \cos \phi  \\\\
v &=& r \sin \phi  \\\\
r &=& \sqrt{u^2 + v^2} \\\\
\phi &=& \tan^{-1}(v/u)
\end{eqnarray}$$


In our Complex number package we want to support the following arithmetic operations on pairs of complex numbers: `add`, `sub`, `mult`, and `div`.

When adding or subtracting complex numbers it is natural to work with their rectangular coordinates:

$$
\begin{eqnarray}
Re(z_1 + z_2) &=& Re(z_1) + Re(z_2) \\\\
Im(z_1 + z_2) &=& Im(z_1) + Im(z_2)
\end{eqnarray}
$$

While when multiplying and dividing complex numbers it is more natural to work with the polar coordinates:

$$
\begin{eqnarray}
Magnitude(z_1 \cdot z_2) &=& Magnitude(z_1) \cdot Magnitude(z_2) \\\\
Angle(z_1 \cdot z_2) &=& Angle(z_1) + Angle(z_2)
\end{eqnarray}
$$

The product is the vector obtained by stretching the length of $z_1$ by the length of $z_2$, and rotating the angle of $z_1$ by the angle of $z_2$.

So there are two different representations which are suitable for different operations. However we want to be able to do all the arithmetic operations on complex numbers regardless of which representation is used.

### Rectangular Representation

 How can we model this number pair using the tools in Clojure? Clojure allows us to create an object called a `Record`. A `Record` is basically a map with a name, a minimum set of keys that it is guaranteed to have, and a constructor. Here's how we could create a rectangular complex number with records:

```clojure
(defrecord Complex-r [real imag])
```

You can create some `Complex-r`'s:

```clojure
=> (->Complex-r 2 3)
#user.Complex-r{:real 2, :imag 3}

=> (->Complex-r -1 16)
#user.Complex-r{:real -1, :imag 16}
```

With records it is good practice to create your own constructor to give you freedom to add post- and pre-conditions when a new record is created. This is just a wrapper function:

```clojure
(defn complex-r
  "create a new Complex-r"
  [re im]
  {:pre [(number? re)
         (number? im)]}
  (->Complex-r re im))
```

I provided some preconditions that assert that the parameters are a type of Clojure's native number. We can access the real and imaginary parts of a `Complex-r`:

```clojure
=> (:real (complex-r 4 5))
4
=> (:imag (complex-r 4 5))
5
```

However, getting the real and imaginary parts of a complex number using the keywords seems to expose the implementation of `Complex-r` too much. Better practice would be to wrap those in some functions:

```clojure
(defn real-r
  "Get the real part of a complex-r number."
  [z]
  (:real z))
  
(defn imag-r
  "Get the imaginary part of a complex-r number."
  [z]
  (:imag z))
```

So now we have:

```clojure
=> (real-r (complex-r 4 5))
4
=> (imag-r (complex-r 4 5))
5
```

We can also view the magnitude and angle of a rectangular complex number using the formulae above:

```clojure
(defn magnitude-r
  "Magnitude of a complex-r number"
  [z]
  (Math/sqrt (+ (square (real-r z))
                (square (imag-r z)))))
                     
(defn angle-r
  "Angle of a complex-r number"
  [z]
  (Math/atan (/ (imag-r z) (real-r z))))
```

In the REPL:

```clojure
=> (magnitude-r (complex-r 3 4))
5.0
=> (angle-r (complex-r 3 4))
0.9272952180016121
```


### Polar Representation

Similarly, we can implement the Polar form of complex numbers as another record:

```clojure
(defrecord Complex-p [magnitude angle])

(defn complex-p
  "create a new Complex-p"
  [magnitude angle]
  {:pre [(number? magnitude)
         (number? angle)]}
  (->Complex-p magnitude angle))
```

Again for the Polar representation we need to write some functions that will give us real and imaginary parts, and the magnitude and angle of a polar number.

```clojure
(defn real-p
  "real part of a complex-p number"
  [z]
  (* (:magnitude z) (Math/sin (:angle z))))
  
(defn imag-p
  "imaginary part of a complex-p number"
  [z]
  (* (:magnitude z) (Math/cos (:angle z))))
  
(defn magnitude-p
  "magnitude of a complex-p number"
  [z]
  (:magnitude z))
  
(defn angle-p
  "angle of a complex-p number"
  [z]
  (:angle z))
```

In the REPL:

```clojure
=> (angle-p (complex-p 5 0.5))
0.5
=> (magnitude-p (complex-p 5 0.5))
5
=> (real-p (complex-p 5 0.5))
4.387912809451864
=> (imag-p (complex-p 5 0.5))
2.397127693021015
```


## Single Dispatch with Protocols

At this point we have two different types of complex number representations and two sets of functions that are specialised to handle each type. This is obviously bad because a user of this numbers module has to pay attention at all times to whether they are using `Complex-r` or `Complex-p` types. They need to specialise whatever code they write with them.
 
Rather than having `real-r` and `real-p` functions we instead want to have a single function `real` that takes any type of complex number and performs _dispatch_ at runtime based on the type of the argument it has received. I.e. dispatch based on the type of the first argument passed to the function. Dispatch based on a single argument is called _single dispatch_.

Clojure allows single dispatch through __Protocols__. A protocol is a named set of functions and their signatures, with no implementions. The functions dispatch on the type of their first argument, and thus must have at least one argument. Implementations of the protocol functions have to be written for each type implementing the protocol. They are very similar to Java _interfaces_, but with one important improvement: which protocols are implemented for a type is not a design time decesion by the code author, rather protocols can extend any type whenever and wherever you want.

We create a protocol for complex numbers using `defprotocol`:

```clojure
(defprotocol PComplex
  (real [z] "Real part of a complex number")
  (imag [z] "Imaginary part of a complex number")
  (magnitude [z] "Magnitude of a complex number")
  (angle [z] "Angle of a complex number"))
```

Implement the `PComplex` protocol for each of our types:

```clojure
(extend-protocol PComplex
  ;; implementation of methods for Complex-r type
  Complex-r
  (real [z] (:real z))
  (imag [z] (:imag z))
  (magnitude [z]
    (Math/sqrt (+ (square (real z))
                  (square (imag z))))             
  (angle [z]
    (Math/atan (/ (imag z) (real z))))  
    
  ;; implemention of methods for Complex-p type
  Complex-p
  (real [z]
    (* (:magnitude z) (Math/sin (:angle z))))
  (imag [z]
    (* (:magnitude z) (Math/cos (:angle z))))
  (magnitude [z] (:magnitude z))
  (angle [z] (:angle z)))
```

Trying this out in a REPL:

```clojure
=> (def z1 (complex-r 5 6))
=> (def z2 (complex-p 3 1))
=> (real z1)
 5
=> (real z2)
 1.6209069176044193
=> (magnitude z1)
 7.810249675906654
=> (magnitude z2)
 5
```

With protocol we now have a generic set of functions for dealing with any type of complex number. If we created a new type of complex number then we'd simple make it implement the `PComplex` protocol.


## Arithmetic With Complex Numbers


The `PComplex` protocol allows us to write code that works with complex numbers and does not need to worry whether whether they are rectangular or polar. We can now write single implementations the arithmetic functions `add`, `sub`, `mult`, and `div` using the formulas above.

```clojure
(defn add
  "Add two complex numbers together"
  [z1 z2]
  (complex-r (+ (real z1) (real z2))
             (+ (imag z1) (imag z2)))
             
(defn sub
  "Subtract two complex numbers from each other"
  [z1 z2]
  (complex-r (- (real z1) (real z2))
             (- (imag z1) (imag z2))))
             
(defn mult
  "Multiply two complex numbers together"
  [z1 z2]
  (complex-p (* (magnitude z1) (magnitude z2))
             (+ (angle z1) (angle z2))))
     
(defn div
  "Divide two complex numbers by each other"
  [z1 z2]
  (complex-p (/ (magnitude z1) (magnitude z2))
             (- (angle z1) (angle z2)))) 
```

Since the functions from `PComplex` work transparently for both representations of complex numbers we only need to write one function for `add` and it works not only for both types, but also all combinations of them for free!

Let's try it all out in the REPL:

```clojure
=> (add (complex-r 5 5) (complex-r 6 6))
#numbers.core.Complex-r{:real 11, :imag 11}
=> (mul (complex-r 5 5) (complex-r 5 5))
#numbers.core.Complex-p{:magnitude 50.00000000000001, :angle 1.5707963267948966}
=> (add (complex-p 5 0.1) (complex-p 10 0))
#numbers.core.Complex-r{:real 14.975020826390129, :imag 0.4991670832341408}
=> (mul (complex-p 5 0.1) (complex-p 10 0))
#numbers.core.Complex-p{:magnitude 50, :angle 0.1}
=> (add (complex-r 5 5) (complex-p 10 0))
#numbers.core.Complex-r{:real 15.0, :imag 5.0}
=> (mul (complex-r 5 5) (complex-p 10 0))
#numbers.core.Complex-p{:magnitude 70.71067811865476, :angle 0.7853981633974483}
```


Without the polymorphism obtained from the protocol we would have to write 16 separate functions instead of just these 4. Moreover, if we wanted to create more complex number representations there would be a combinatorial explosion in the number of arithmetic functions we'd need to write.

<br />
_[> Click here for the next part of this series.]({% post_url 2016-11-21-polyclojure2 %})_
