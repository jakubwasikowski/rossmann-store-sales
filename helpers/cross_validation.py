def non_random_train_test_split(data_set, test_size=0.3):
    if not 0 < test_size < 1:
        raise Exception("Test size should be between 0 and 1")
    split_idx = int(test_size * len(data_set))
    return data_set[split_idx:], data_set[:split_idx]
