import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox


def load_dataset(data_filename, target_filename):
    """
    Load the dataset from 2 files - the data and the target.
    Concatenate them into a single data frame.
    """
    data_train = pd.read_csv(data_filename, delimiter='\t', header=0)
    data_target = pd.read_csv(target_filename, delimiter='\t', header=None)
    data = pd.concat([data_train, data_target], axis=1)

    data.rename(columns={0: 'target'}, inplace=True)

    return data


def load_dataset_target(data_filename, target_filename):
    """
    Load the dataset from 2 files. This time, don't merge them,
    return them separately instead.
    """
    data_train = pd.read_csv(data_filename, delimiter='\t', header=0)
    data_target = pd.read_csv(target_filename, delimiter='\t', header=None)

    return data_train, data_target


def preprocess_data(data):
    """
    Preprocess a dataset. Reduce the number of features, normalize the rest,
    and create new useful features.
    """
    described_data = data.describe().transpose()
    single_value_columns = described_data[described_data['std'] == 0].index

    data = data.drop(single_value_columns, axis=1)

    nunique = data.apply(pd.Series.nunique)
    few_nunique = nunique[(nunique < 5) & (nunique.index != 'proxy')].index

    data = data.drop(few_nunique, axis=1)

    data_transformed = data.drop(['proxy', 'target'], axis=1).apply(lambda x: boxcox(x + 1)[0], raw=True)
    data_norm = pd.DataFrame(StandardScaler().fit_transform(data_transformed))
    data_norm.columns = data.drop(['proxy', 'target'], axis=1).columns
    data_norm[['proxy', 'target']] = data[['proxy', 'target']]

    return data_norm

