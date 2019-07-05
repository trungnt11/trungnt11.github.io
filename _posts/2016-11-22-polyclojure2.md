---
layout: post
title: "Polymorphism in Clojure:<br>A Tutorial Using Numbers (2/3)"
date: 2016-11-22
comments: false
tags: clojure polymorphism functional-programming
---

_This post is composed of 3 parts: [Part 1]({% post_url 2016-11-21-polyclojure1 %}), Part 2, [Part 3]({% post_url 2016-11-23-polyclojure3 %})_

## More Number Types

Let's extend our number package with more number types: An Integer type, a Float type for real numbers, and a Rational type for fractions. Like before we create the records and constructors:

```clojure
(defrecord Int [n])
(defrecord Rational [n d])
(defrecord Float [n])

;; constructors
(defn integer
  "Make a new integer"
  [n]
  {:pre [(number? n)]}
  (->Int (int n)))
  
(defn float
  "Make a new float"
  [n]
  {:pre [(number? n)]}
  (->Float (double n)))
  
(defn rational
  "Make a new rational"
  [n d]
  {:pre [(number? n)
         (number? d)]}
  (let [g (gcd n d)]
    (->Rational (/ n g) (/ d g)))) 
```

Trying these out in the REPL:

```clojure
=> (float 3)
#numbers.core.Float{:n 3.0}

=> (integer 6)
#numbers.core.Int{:n 4}

=> (rational 6 3)
#numbers.core.Rational{:n 2, :d 1}
```

## Generic Arithmetic

We want to create an `add` function that can be called with either two integers, two rationals, two reals, or two complex types and do the right thing in every case. Protocols in Clojure allow for _single dispatch_ only. Here we need to dispatch on the types of _multiple_ arguments.

We could hack this with a mega-function that is just a big conditional statement:

```clojure
(def mega-add
   "one add to rule them all (don't do this)"
   [n1 n2]
   (cond
     (and (= (type n1) Int)
          (= (type n2) Int))
       (add-int n1 n2)
     (and (= (type n1) Float)
          (= (type n2) Float)
       (add-real n1 n2)
     ;; etc
     :else (throw "unknown types")))
```

The problem with this solution is that it is _closed for business_. If a user of the our number library desired to extend the number system with a new type of number (e.g. a BigInt), they'd have to break in and edit this function directly.


### Multi-Methods

Clojure's core library provides multiple dispatch via __multi-methods__. While protocols in Clojure perform single-dispatch on just the type of the first argument, multi-methods are much more general and allow the programmer to define their own rules for dispatch using any number of arguments. You are not limited to dispatch with just the types of the arguments, but also their values.

Let's throw out `mega-add` and do it properly with  multi-methods. The multi-method is defined using the  `defmulti` macro. It takes a docstring and a dispatch function as its arguments. For adding, the dispatch function will be mapped to the two numbers as arguments and so return a vector of the types of the  arguments:

```clojure
(defmulti add
  "Generic add"
  class)
```

So if we provided two `Int`s then the dispatch would return `[Int Int]`. With the dispatch machinery is in place, we now need to add the implementations for each of the types. This is done with `defmethod`, which defines a method for each valid output of the dispatch function:

```clojure
(defmethod add [Int Int]
  "Add two integers together"
  [n1 n2]
  (integer (+ (:n n1) (:n n2))))
  
(defmethod add [Float Float]
  "Add two floats together"
  [n1 n2]
  (float (+ (:n n1) (:n n2))))
  
;; etc
```

Trying this out in the repl:

```clojure
=> (add (integer 4) (integer 6))
#numbers.core.Int{:n 10}

=> (add (float 5) (float 10))
#numbers.core.Float{:n 15.0}
```

Neat! Multi-methods are easy to extend too. If I wanted to create a new number type (e.g. a BigInt), then all I need to do is add a new method with `defmethod` for the case of `[BigInt BigInt]`.

Similarly we can reimplement the `add` function defined previously for the two complex number types, using the new multi-method machinery:

```clojure
(defmethod add [Complex-r Complex-r]
  "Add two complex-r numbers together"
  [z1 z2]
  (complex-r (+ (real z1) (real z2))
             (+ (imag z1) (imag z2)))

(defmethod add [Complex-p Complex-p]
  "Add two complex-p numbers together"
  [z1 z2]
  (complex-r (+ (real z1) (real z2))
             (+ (imag z1) (imag z2)))
```

This works fine, but previously `add` for the two different complex number types was a single function, whereas now we have repetition. Moreover we can't add a `Complex-r` to a `Complex-p` like we could before.

Multimethods have provided a lot of extensibility to new number types, but at the same time we have lost the polymorphic nature we had in the arithmetic functions of the two complex types. I will address this problem in the next section.


## Keywords and Dispatch Hierarchies

We have an impression that `Complex-r` and `Complex-p` are subtypes of some imaginary abstract type `Complex`. However Clojure does not offer any notion of an 'abstract type' as we are used to in OOP. Instead Clojure provides an ad-hoc dynamic hierarchy system. The hierarchy system supports derivation relationships between names (either symbols or keywords), and relationships between classes and names. 

The `derive` function creates these relationships, and the `isa?` function tests for their existence. We will use namespaced keywords (double colon) to represent the number types:

```clojure
=> (derive ::complex-r ::complex)
=> (derive ::complex-p ::complex)

=> (isa? ::complex-r ::complex)
true
=> (isa? ::complex-p ::complex)
true
=> (isa? ::complex-r ::complex-p)
false

=> (ancestors ::complex-r)
#{::complex}
=> (ancestors ::complex-p)
#{::complex}
```

What we want to do is rewrite the arithmetic multi-methods to dispatch using these namespaced keywords in place of the number types. The complex `add` method could then be reduced to matching arguments that satisfy: `[::complex ::complex]`.
To do this, we will require a one-to-one mapping of each type to its associated keyword: 

* `Complex-r` => `::complex-r`
* `Complex-p` => `::complex-p`
* `Float` => `::float`
* `Rational` => `::rational`
* `Int` => `::integer`

We could do this with a global lookup table or add the keywords to the record definitions, but these are cludgy solutions. The first requires maintaining some global data, and the second repeats information and forces us to rewrite the record definition, which would break existing code. A cleaner solution is just to create another __protocol__ and extend our number types with it:

```clojure
(defprotocol PNumberKind
  (kind [n] "The keyword name for the kind of the number n"))
   
(extend-protocol PNumberKind
  Complex-r
  (kind [z] ::complex-r)
  
  Complex-p
  (kind [z] ::complex-p)
  
  Float
  (kind [z] ::float)
  
  Rational
  (kind [z] ::rational)
  
  Int
  (kind [z] ::integer))
``` 

In the REPL:

```clojure
=> (kind (integer 3))
:numbers.core/integer

=> (kind (complex-r 4 5))
:numbers.core/complex-r
```

We can now update the dispatch function used by the multimethod to dispatch using `kind`:

```clojure
(defmulti add
  "Generic add"
  kind)
```

The methods can now be rewritten as:

```clojure
(defmethod add [::integer ::integer]
  "Add two integers together"
  [n1 n2]
  (integer (+ (:n n1) (:n n2))))
  
(defmethod add [::float ::float]
  "Add two reals together"
  [n1 n2]
  (real (+ (:n n1) (:n n2))))
  
(defmethod add [::complex ::complex]
  "Add two complex-p numbers together"
  [z1 z2]
  (complex-r (+ (real z1) (real z2))
             (+ (imag z1) (imag z2)))
```

Since we added the rule `(derive ::complex-r ::complex)` to the hierarchy, the multimethod called with `::complex-r` or `::complex-p` implicitly satisfy the rule `[::complex ::complex]`. The hierarchy has therefore allowed us the implement a polymorphic add for adding different representations of complex numbers and their combinations. If we added more complex number representations, the generic add method for complex numbers would accomodate them automatically without modification.

Let's try this in the REPL:

```clojure
=> (add (complex-r 2 3) (complex-r 6 7))
#numbers.core.Complex-r{:real 8, :imag 10}

=> (add (complex-p 4 5) (complex-p 1 0))
#numbers.core.Complex-r{:real 2.1346487418529048, :imag -3.835697098652554}

=> (add (complex-p 3 4) (complex-r 5 6))
#numbers.core.Complex-r{:real 3.039069137409164, :imag 3.7295925140762156}
```

<br />         
_[> Click here for the next part of this series.]({% post_url 2016-11-23-polyclojure3 %})_

_[> Click here for the previous part of this series.]({% post_url 2016-11-21-polyclojure1 %})_