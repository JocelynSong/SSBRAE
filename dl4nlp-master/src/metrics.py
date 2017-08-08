def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    import sklearn.metrics
    return sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred,
                                          normalize=normalize, sample_weight=sample_weight)