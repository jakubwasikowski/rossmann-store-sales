import csv

from features.feature_set import FeatureSet


class StoreState(FeatureSet):
    def __init__(self, state_per_store_path):
        FeatureSet.__init__(self)
        self._initialize_states(state_per_store_path)

    def generate_features(self, data_set, features):
        features["state"] = data_set.Store.apply(lambda x: self.state_per_store[x]).astype("category")

    def _initialize_states(self, store_states_path):
        self.state_per_store = {}
        with open(store_states_path) as store_states_file:
            store_states_reader = csv.reader(store_states_file)
            store_states_reader.next()
            for store_id, state_code in store_states_reader:
                self.state_per_store[int(store_id)] = state_code
