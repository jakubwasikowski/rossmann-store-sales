from collections import OrderedDict
import pandas as pd
from scipy import sparse


class FeaturesExtractor:
    def __init__(self):
        self._feature_set_generators = []

    def add_feature_set(self, feature_set):
        self._feature_set_generators.append(feature_set)

    def extract(self, data_set, feature_names=None):
        feature_set = self._generate_feature_set(data_set)
        return self._to_sparse_structure(feature_set, feature_names)

    def _generate_feature_set(self, data_set):
        print "Generating features..."
        feature_set = pd.DataFrame(index=data_set.index)
        for fs_gen in self._feature_set_generators:
            fs_gen.generate_features(data_set, feature_set)
        return feature_set

    def _to_sparse_structure(self, feature_set, feature_names):
        print "Converting to sparse structure..."
        categorical_columns, quantitative_columns = self._get_columns(feature_set)

        row = []
        col = []
        data = []
        if feature_names is not None:
            features_indexes = OrderedDict((fn, i) for i, fn in enumerate(feature_names))
        else:
            features_indexes = OrderedDict()

        rows_no = 0
        for fs_index, fs_row in feature_set.iterrows():
            if (rows_no + 1) % 10000 == 0:
                print "Percentage of converted rows: %.1f%%" % (float(rows_no) * 100 / len(feature_set))
            for q_col in quantitative_columns:
                self._update_feature_indexes(features_indexes, q_col)
                row.append(rows_no)
                col.append(features_indexes[q_col])
                data.append(fs_row[q_col])
            for c_col in categorical_columns:
                one_hot_encoded_c_col = "%s=%s" % (c_col, str(fs_row[c_col]))
                self._update_feature_indexes(features_indexes, one_hot_encoded_c_col)
                row.append(rows_no)
                col.append(features_indexes[one_hot_encoded_c_col])
                data.append(1)
            rows_no += 1

        sparse_features = sparse.csr_matrix((data, (row, col)), shape=(rows_no, len(features_indexes)))
        features_names = features_indexes.keys()
        return sparse_features, features_names

    def _update_feature_indexes(self, features_indexes, column):
        if column not in features_indexes:
            features_indexes[column] = len(features_indexes)

    def _get_columns(self, feature_set):
        categorical_columns = []
        quantitative_columns = []
        for column in feature_set.columns.values:
            if feature_set.dtypes[column].name == "category":
                categorical_columns.append(column)
            else:
                quantitative_columns.append(column)

        return categorical_columns, quantitative_columns
