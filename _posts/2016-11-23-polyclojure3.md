---
layout: post
title: "Polymorphism in Clojure:<br>A Tutorial Using Numbers (3/3)"
date: 2016-11-23
comments: false
tags: clojure polymorphism functional-programming
---

_This post is composed of 3 parts: [Part 1]({% post_url 2016-11-21-polyclojure1 %}), [Part 2]({% post_url 2016-11-22-polyclojure2 %}), Part 3_


## Arithmetic with Mixed Types

We have so far built a number system with five different types and an `add` function that can take any two numbers of the same type and add them together with the same interface. 

However what if I wanted to `add` a `Complex-r` and a `Real` together? We would need to convert the `Real` to a `Complex-r` and then add them together. We could do this by adding more multimethods:

```clojure
(defmethod add [::complex-r ::real]
  (...))
  
(defmethod add [::real ::complex-r]
  (...))
```

You can see that doing this for all combinations of types would lead to a combinatorial explosion of new multimethods! Clojure's hierarchies can enable us to solve this problem without the need to write a factorial number of functions.

We can side-step the problem of writing a combinatorical number of new functions by noting that to add two different types of number together we have to promote one of the numbers to be the same type as the other. With this in mind we can create a catch-all method for `add` that catches all the cases of mixed types, coerces the types to be the same and then calls `add` again with the converted types. It will then call one of the previously defined `add` methods where the types are the same. 

We can implement the catch-all case by creating a new abstraction `::number` at the top of the hierarchy where every number we have created so far is a descendant of `::number`:

```clojure
(derive ::complex ::number)
(derive ::integer ::number)
(derive ::float ::number)
(derive ::rational ::number)
```

We can now create the catch-all `add` method that will that will matc `[::number ::number]`:

```clojure
(defmethod add [::number ::number]
  [n1 n2]
  (apply add (coerce-types n1 n2)))
```

This works because any combination of different types will fail all the rules of the other multimethods except this one, because all of our types are descendants of `::number`. This method then calls the coercion function (which I will define later) to convert the arguments into the same type and then calls the `add` multi-function with these converted arguments. This will then find the correct multimethod for the now uniform types return the result. 

## Type Coercion with the Numeric Tower

We can convert any Integer to a Rational without a loss of information. You cannot convert any Rational to an Integer without a loss of information though. Similarly any Rational can be converted to a Float, and any Float can be converted to a Complex number.

This chain of conversions is the __numeric tower__:

* Integer -> Rational -> Float -> Complex   

We want to be able to call a function `raise` on one of our types and get back the same numeric value, but represent by the next type in the tower. The function depends on the type of the argument so we can create another protocol:

```clojure
(defprotocol PRaise
  (raise [x] "raises a number type to the next type
          in the numeric tower with equivalent value"))

(extend-protocol PRaise
  Int
  (raise [x] (rational (:n x) 1))

  Rational
  (raise [x] (float (/ (numer x) (double (denom x)))))

  Float
  (raise [x] (complex-r (:n x) 0)))
```

Given a pair of types, e.g. `[::integer ::float]`, we need a way to encode the fact that `::float` is higher in the tower than `::integer`, and so `::integer` needs to be raised until it is a `::float`. This could be done with a map associating the types to a rank number, but this is pretty inflexible if we add more types. 

The numeric tower is a hierarchy so we are actually better off using Clojure's ad-hoc hierarchies again. In Clojure there is a global hierarchy structure which we have used so far for the arithmetic, but you are free to create your own hierarchies with `make-hierarchy`. This exists independently of the global hierarchy we used earlier. This is handy because the numeric tower hierarchy is different from the arithmetic hierarchy.

```clojure
(def numeric-tower
  (-> (make-hierarchy)
      (derive ::complex ::complex-p)
      (derive ::complex ::complex-r)
      (derive ::float ::complex)
      (derive ::rational ::float)
      (derive ::integer ::rational)))
```

Using this we can create comparator functions for two keyword types. For example:

```clojure
=> (higher? ::float ::integer)
true
=> (lower? ::rational ::complex-r)
true
```

These are easy to implement using the functions for querying hierarchies, `ancestors` and `descendents`:

```clojure
(defn higher?
  "Is type 1 higher in the numeric tower than type 2?"
  [t1 t2]
  (if (contains? (ancestors numeric-tower t2) t1)
    true
    false))

(defn lower?
  "Is type 1 lower in the numeric tower than type 2?"
  [t1 t2]
  (if (contains? (descendants numeric-tower t2) t1)
    true
    false))
```

With these functions we can implement `coerce-types`:

```clojure
(defn coerce-types
  "Given two different number types raise the lesser type up
  to be the same as the greater type"
  [x y]
  (let [t1 (kind x)
        t2 (kind y)]
    (cond
      (lower? t1 t2) (recur (raise x) y)
      (higher? t1 t2) (recur x (raise y))
      :else [x y]))) 
```

Trying this out in a REPL:

```clojure
=> (raise (integer 3))
#numbers.core.Rational{:n 3, :d 1}

=> (raise (float 4))
#numbers.core.Complex-r{:real 4.0, :imag 0}

=> (coerce-types (integer 4) (rational 5 6))
[#numbers.core.Rational{:n 4, :d 1} #numbers.core.Rational{:n 5, :d 6}]

=> (coerce-types (rational 5 6) (complex-r 7 8))
[#numbers.core.Complex-r{:real 0.8333333333333333, :imag 0} #numbers.core.Complex-r{:real 7, :imag 8}]
```

## Final Product and Further Ideas

We have implemented a number system that can represent integers, floating point numbers, rational numbers, rectangular complex numbers, and polar complex numbers. It can perform basic binary arithmetic operations add, subtract, multiply, and divide on any combination of number types.

Let's demonstrate the final product in the REPL:

```clojure
(add (integer 4) (integer 6))
#numbers.core.Int{:n 10}

(add (integer 4) (float 6))
#numbers.core.Float{:n 10.0}

(add (rational 5 6) (float 6))
#numbers.core.Float{:n 6.833333333333333}

(mul (float 6) (complex-p 10 0.5))
#numbers.core.Complex-p{:magnitude 60.0, :angle 0.5}

(div (integer 5) (rational 7 5))
#numbers.core.Rational{:n 25, :d 7}

(div (integer 5) (complex-r 7 5))
#numbers.core.Complex-p{:magnitude 0.5812381937190965, :angle -0.6202494859828215}
```

Seems to work fairly well!

I could continue building on this, but that would be beyond the scope of this blog post. There are a few things worth thinking about nonetheless. One aesthetic improvement would be implementing a pretty REPL representation of the custom number types. E.g. having `3/4` instead of `Rational(:n 3 :d 4)`. Clojure has a single function that takes care of printing things: `clojure.lang.print-method`. This is simply another multimethod like we've already been using. Adding nicer printing is straight-forward:

```clojure
;; nicer printing for rationals
(defmethod print-method Rational [x ^java.io.Writer writer]
  (.write writer (str (numer x) \/ (denom x))))
  
;; and similarly for the other types...
;; ...
```

In SICP there is also an exercise to extend the number system to be implemented purely with its own types. So while Integer and Float types would be wrappers for Clojure's own types, the Rational, Complex-r, and Complex-p types could themselves be composed of any of the other types. So you could have a Rational number whose demoninator and numerator were complex numbers. Or conversely a complex number whose real and imaginary parts were rational numbers. This would be achieved with some modification to the existing code. You would need to replace all uses of Clojure's primitive arithmetic functions (`+`, `*`, etc.) with our multimethods (`add`, `mult`, etc), and also create our own implementations of `sqrt`, `sin`, `cos`, and `atan` that handled our number types. This could be done by adding new protocols. 



## Final Thoughts

This is a nice non-trivial program from SICP that demonstrates the ideas and challenges in polymorphism. Implementing this in Clojure required us to use just about every feature for polymorphism in Clojure's core library. But I am satified with how Clojure could handle everything without any requiring any 'hacks' or major redesigns.

I think that this highlights one of the good aspects of Clojure's design and philosophy. Namely, the decoupling (or 'decomplecting') of ideas in the language. In this demo there are the following concepts:

1. Data (in the form of records).
2. Functions.
3. Single and multiple dispatch (protocols and multimethods) to functions.
4. Hierarchies of types.


Functions and data are decoupled because records are just data and you can't attach methods onto records in Clojure the way you can with class methods in OOP. You don't need getter/setter methods because records are maps so you just the functions for maps. 

Data and dispatch are decoupled. You do not need to know at design time which protocols or multimethods are going to use your record type. In Java where polymorphic single-dispatch is achieved via interfaces or abstract classes you need explicitly implement or extend your class when you write it. In Clojure you can add protocols or multimethods to any existing type when are where you want.

Dispatch in Clojure is simply a fancy wrapper for functions. For a user of a protocol, multimethod, or function it makes no difference how it is implemented - it looks exactly the same. This is a big win for extending or refactoring code without breaking things. For example, say we only had a complex type `complex-r` and implemented the methods `real`, `imag`, `magnitude`, and `angle` as functions via `defn`. But then later we required the `complex-p` type and it also needed the same methods. Refactoring the existing functions (`real`, `imag`, etc) into a protocol will make no difference to code already using these functions with the `complex-r` type - it looks and acts just like a function. Multimethods are the same. 

Type hierarchies can be decoupled from types. In this example we built a hierarchy for the number types using namespaced keywords like `::complex-r`. This exists independently from the records we defined. It is coupled to the record types via a one-to-one mapping of the records to a keyword, implemented by the protocol `PNumberKind`. This decoupling allowed us to create further abstractions such as `::number` and `::complex`. In Java you'd have to create abstract base classes retroactively and subclass the existing types, which would be a redesign. This decoupling was also useful later for the _numeric tower_ where we actually required a completely different hierarchy of the types. This was made possible in Clojure because you can create multiple hierarchies ad-hoc - hierarchies are just data. You could even implement multimethods with different type hierarchies this way. In OOP this kind of polymorphic dispatch is strongly coupled to your class hierarchies, which you can't just change.


## Source Code

The complete code of this tutorial can be found here: https://github.com/jimypbr/clojure-numbers.

## Further Reading

1. Read SICP section 2.4: https://mitpress.mit.edu/sicp/full-text/book/book-Z-H-17.html#%_sec_2.4
2. Watch 'Simple Made Easy' by Rich Hickey. It's a great talk that explains how Clojure aims to decouple key programming concepts from each other: https://www.infoq.com/presentations/Simple-Made-Easy
3. For more about multiple dispatch, Eli Bendersky's series of blog posts are great: http://eli.thegreenplace.net/2016/a-polyglots-guide-to-multiple-dispatch


<br />         
_[> Click here for the previous part of this series.]({% post_url 2016-11-22-polyclojure2 %})_
