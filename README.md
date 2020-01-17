``slim-python`` is a package to create scoring systems.

## Introduction

[SLIM](http://http//arxiv.org/abs/1502.04269/) is new machine learning method to learn *scoring systems* -- binary classification models that let users make quick predictions by adding, subtracting and multiplying a few small numbers.


SLIM can learn models that are fully optimized for accuracy and sparsity, and that satisfy difficult constraints **without parameter tuning** (e.g. hard limits on model size, the true positive rate, the false positive rate).

## Requirements

``slim-python2`` was developed using Python 2.7.11 and CPLEX 12.6.2. It may work with other versions of Python and/or CPLEX, but this has not been tested and will not be supported in future releases.

``slim-python3`` was developed using Python 3.7.3 and ortools.

## Citation

[SLIM paper](http://http//arxiv.org/abs/1502.04269/)  

```
@article{
    ustun2015slim,
    year = {2015},
    issn = {0885-6125},
    journal = {Machine Learning},
    doi = {10.1007/s10994-015-5528-6},
    title = {Supersparse linear integer models for optimized medical scoring systems},
    url = {http://dx.doi.org/10.1007/s10994-015-5528-6},
    publisher = { Springer US},
    author = {Ustun, Berk and Rudin, Cynthia},
    pages = {1-43},
    language = {English}
}
```
