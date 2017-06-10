Cost-sensitive multiclass classification with Adaptive Regularization of Weights

This is a modification of Andreas Vlachos' original cost-sensitive, multiclass AROW code here: https://github.com/andreasvlachos/arow_csc

Authors:
* Andreas Vlachos
* Gerasimos Lampuras
* Johann Petrak

This contains the following changes:
* better command-line interface
* pluggable back-ends for the sparse feature vectors
* works with Python2 and Python3
* (TODO) possible to use like a sklearn model
* (TODO) can deal with various data formtats, including vowpal\_wabbit

## Representation of Feature Vectors

The program internally represents the attributes of each instance and
the weight vectors as sparse vectors. The program supports the use
of different implementations for representing those sparse vectors
(this can be selected with the -F option of the arow_csc.py program):

* defaultdict: this is the default. It works with Python2 and Python3, but
  is very slow.
* hvectors: this uses Liang Huang's hvector library (see http://web.engr.oregonstate.edu/~huanlian).
  This is very fast but only works with Python 2.x
* sv: this uses the sparsevectors library (see https://github.com/johann-petrak/python-sparsevectors).
  This is fast but only works with Python 3.x and is not thoroughly tested yet.

A comparison of all three approaches is included further down in the "Benchmarks" section.

## Command Line Usage

TBD

## Library Usage

```python
from arow_csc import AROW
model = new AROW()
## TODO: explain X, Y, C
model.train(X,Y,C)
model.train(X2,Y2,C2) ## can incrementally add new
# TBD
```

## Benchmarks

On 1000 examples from news20.

| Python | Backend     | Time(s) | Avg. Test Cost |
| :----- | :------     | ------: | ------------:  |
| 2.7    | defaultdict | 599.7   | 0.128514056225 |
| 2.7    | hvector     | 15.3    | 0.128514056225 |
| 3.5    | defaultdict | 553.2   | 0.164658634538 |
| 3.5    | sv          | 15.8    | 0.160642570281 |
| 3.5    | sv          | 16.1    | 0.160642570281 |
