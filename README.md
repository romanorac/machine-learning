# Machine Learning algorithms

The main intention of this repository is to show simple implementations of machine learning algorithms and metrics. Packages, like numpy,
contain state-of-the-art implementations, but are hard to understand when studying algorithms.

Repository contains following algorithms:
- linear regression,
- locally weighted linear regression,
- decision tree,
- random forest,
- label spreading.

Measures and metrics:
- information gain,
- radial basis function,
- Gower dissimilarity,
- Levenshtein distance.

With regression algorithms, we build a model, predict and plot the data. We show how model changes with different parameters.

With ensemble algorithms, we compare implementations of decision tree and random forest with scikit's algorithms.
We show that algorithms achieve comparable accuracies on selected datasets.

With label spreading algorithm, we compare accuracy of our implementation with scikit's and provide additional
Levenshtein distance measure.

Algorithms require python 2.7. To install requirements run: `pip install -r requirements.txt`.


