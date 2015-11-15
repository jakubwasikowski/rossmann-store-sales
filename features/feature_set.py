class FeatureSet:
    def __init__(self):
        pass

    def generate_features(self, data_set, features):
        raise NotImplementedError()

    def _rewrite(self, data_set, features, features_names, data_type):
        for f_name in features_names:
            features[f_name] = data_set[f_name].astype(data_type)