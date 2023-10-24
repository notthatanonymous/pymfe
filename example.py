"""
Basic of meta-features extraction
=================================

This example show how to extract meta-features using standard configuration.
"""


###############################################################################
# Extracting meta-features
# ------------------------
#
# The standard way to extract meta-features is using the MFE class.
# The parameters are the dataset and the group of measures to be extracted.
# By default, the method extracts general, info-theory, statistical,
# model-based and landmarking measures. For instance:

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

# Load a dataset
data = load_iris()
y = data.target
X = data.data

###############################################################################
# Extracting default measures
mfe = MFE(groups = ['info-theory'])
mfe.fit(X, y)
ft = mfe.extract()


print(f"Score: {ft[1][ft[0].index('mut_inf.mean')]}")
