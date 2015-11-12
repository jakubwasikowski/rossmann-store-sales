import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.

    Taken from https://gist.github.com/ramhiser/982ce339d5f8c9a769a0

    Details:

    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df