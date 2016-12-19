import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin


class LowVarianceRemover(BaseEstimator, TransformerMixin):
    """
    This transform removes all features with few unique values.
    It keeps `proxy` because that is a boolean.
    """

    def fit(self, data, target=None):
        nunique = data.apply(pd.Series.nunique)
        few_nunique = nunique[(nunique < 5) & (nunique.index != 'proxy')].index

        self.toremove = few_nunique
        return self

    def transform(self, data):
        return data.drop(self.toremove, axis=1)


class BoxcoxTransform(BaseEstimator, TransformerMixin):
    """
    Applies box-cox transform to all features. Specify ignored features
    in the constructor.
    """

    def fit(self, data, target=None):
        # self.mins = [min(x) for x in data]
        self.alphas = pd.DataFrame(data).apply(lambda x: boxcox(x - min(x) + 1)[1], raw=True)
        return self

    def transform(self, data):
        return pd.DataFrame(data).apply(lambda x: boxcox(x.values - min(x.values) + 1, alpha=self.alphas[x.name])[0])

