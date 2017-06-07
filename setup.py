import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return open(os.path.join(here, fname)).read()
readme = read('README')

setup(
    name="arow_csc",
    version="0.2",
    description=("Cost-sensitive multiclass classification with Adaptive Regularization of Weights"),
    author="Andreas Vlachos",
    #author_email = "notyet@somewhere.com",
    #url = "http://packages.python.org/an_example_pypi_project",
    license="BSD",
    #keywords = "example documentation tutorial",
    packages=['arow_csc'],
    long_description=readme,
    py_modules=['arow_csc'],
    scripts=['arow_csc.py'],
    entry_points = {'console_scripts': ['arow_csc=arow_csc:main']},
    tests_require = ['nose'],
    test_suite = 'nose.collector' 
)
