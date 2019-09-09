import numpy as np
import pandas as pd


class Metric(object):
    def __call__(self, y, y_pred, gids=None):
        if gids is None:
            gids = np.zeros(len(y))
        raise NotImplementedError()


def _precision(true_relevance):
    return true_relevance.sum() / len(true_relevance)


def average_precision_at_k(k, y, y_pred):
    y_top = _get_y_top(k, y, y_pred)
    return np.sum([
        y_top[:(ind + 1)].sum() * 1. / (ind + 1)
        for ind in range(len(y_top))
        if y_top[ind] > 0
    ]) / len(y_top)


def _get_y_top(k, y, y_pred):
    y_top = y[np.argsort(y_pred)[:-k - 1:-1]]
    return y_top


def precision_at_k(k, y, y_pred):
    return _get_y_top(k, y, y_pred).sum() * 1. / k


def cumulative_gain(k, y, y_pred):
    return _get_y_top(k, y, y_pred).sum()


def discounted_cumulative_gain(k, y, y_pred):
    y_top = _get_y_top(k, y, y_pred)
    return np.sum(y_top / np.log(2 + np.arange(len(y_top))))


def normalized_discounted_cumulative_gain(k, y, y_pred):
    denominator = discounted_cumulative_gain(k, y, y)
    if denominator == 0:
        return 0
    return discounted_cumulative_gain(k, y, y_pred) / discounted_cumulative_gain(k, y, y)


def reciprocal_rank(k, y, y_pred):
    for ind, y in enumerate(_get_y_top(k, y, y_pred)):
        if y > 0:
            return 1.0 / (ind + 1)
    return 0.


class MeanMetric(Metric):
    def _group_metric(self, y, y_pred):
        raise NotImplementedError()

    def __call__(self, y, y_pred, gids=None):
        return pd.DataFrame({
            "y": y,
            "y_pred": y_pred,
            "gid": gids
        }).groupby("gid").apply(lambda df: self._group_metric(df.y.values, df.y_pred.values)).mean()


class MeanPrecision(MeanMetric):
    def __init__(self, k=10):
        self.k = k

    def _group_metric(self, y, y_pred):
        return precision_at_k(self.k, y, y_pred)

    def __str__(self):
        return "mean_precision@{}".format(self.k)


class MeanAveragePrecision(MeanMetric):
    def __init__(self, k=10):
        self.k = k

    def _group_metric(self, y, y_pred):
        return average_precision_at_k(self.k, y, y_pred)

    def __str__(self):
        return "mean_average_precision@{}".format(self.k)


class DCG(MeanMetric):
    def __init__(self, k=10):
        self.k = k

    def _group_metric(self, y, y_pred):
        return discounted_cumulative_gain(self.k, y, y_pred)

    def __str__(self):
        return "mean_dcg@{}".format(self.k)


class NDCG(MeanMetric):
    def __init__(self, k=10):
        self.k = k

    def _group_metric(self, y, y_pred):
        return normalized_discounted_cumulative_gain(self.k, y, y_pred)

    def __str__(self):
        return "mean_ndcg@{}".format(self.k)


class ReciprocalRank(MeanMetric):
    def __init__(self, k=10):
        self.k = k

    def _group_metric(self, y, y_pred):
        return reciprocal_rank(self.k, y, y_pred)

    def __str__(self):
        return "mean_reciprocal_rank@{}".format(self.k)


from sklearn.metrics import roc_auc_score, precision_score


def _binarize(y):
    return 1 * (y > 0)


class TotalAUC(Metric):
    def __call__(self, y, y_pred, *args, **kwargs):
        y = _binarize(y)
        if np.max(y) == np.min(y):
            return 0
        return roc_auc_score(y, y_pred)

    def __str__(self):
        return "total_auc"


class MeanAUC(MeanMetric):
    def __call__(self, y, y_pred, *args, **kwargs):
        return super().__call__(_binarize(y), y_pred, *args, **kwargs)

    def _group_metric(self, y, y_pred):
        if np.max(y) == np.min(y):
            return 0
        return roc_auc_score(y, y_pred)

    def __str__(self):
        return "mean_auc"


class TotalPrecision(Metric):
    def __call__(self, y, y_pred, *args, **kwargs):
        return precision_score(_binarize(y), 1 * (y_pred > _binarize(y).mean()))

    def __str__(self):
        return "total_precision"


class MeanPrecision2(MeanMetric):
    def __call__(self, y, y_pred, *args, **kwargs):
        return super().__call__(_binarize(y), y_pred, *args, **kwargs)

    def _group_metric(self, y, y_pred):
        return precision_score(y, 1 * (y_pred > y.mean()))

    def __str__(self):
        return "mean_precision_simple"


from scipy.stats import kendalltau


class MeanKendallTau(MeanMetric):
    def _group_metric(self, y, y_pred):
        if len(y) <= 1:
            return 0.
        return kendalltau(y, y_pred)[0]

    def __str__(self):
        return "mean_kendal_tau"


class MeanKendallTauAtK(MeanMetric):
    def __init__(self, k=10):
        self.k = k
        if k <= 1:
            raise Exception()

    def _group_metric(self, y, y_pred):
        if len(y) <= 1:
            return 0.
        top_indexes = np.argsort(y_pred)[:-self.k - 1:-1]
        return kendalltau(y[top_indexes], y_pred[top_indexes])[0]

    def __str__(self):
        return "mean_kendal_tau@{}".format(self.k)


from scipy.stats import spearmanr


class MeanSpearmanR(MeanMetric):
    def _group_metric(self, y, y_pred):
        if len(y) <= 1:
            return 0.
        return spearmanr(y, y_pred)[0]

    def __str__(self):
        return "mean_spearman_r"


class MeanSpearmanRAtK(MeanMetric):
    def __init__(self, k=10):
        self.k = k
        if k <= 1:
            raise Exception()

    def _group_metric(self, y, y_pred):
        if len(y) <= 1:
            return 0.
        top_indexes = np.argsort(y_pred)[:-self.k - 1:-1]
        return spearmanr(y[top_indexes], y_pred[top_indexes])[0]

    def __str__(self):
        return "mean_spearman_r@{}".format(self.k)

#
# y = np.array([3, 1, 0, 1, 0, 10, 1, 0, 0])
# y_pred = np.array([0.1, 0.5, 0.1, 0, 1, 0.1, 1, 10, 1])
# gids = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
#
# ltr_metrics = [
#     MeanPrecision(),
#     MeanPrecision(3),
#     MeanPrecision(1),
#     MeanAveragePrecision(),
#     MeanAveragePrecision(3),
#     MeanAveragePrecision(1),
#     DCG(),
#     DCG(3),
#     DCG(1),
#     NDCG(),
#     NDCG(3),
#     NDCG(1),
#     ReciprocalRank(),
#     ReciprocalRank(3),
#     ReciprocalRank(1),
#     MeanAUC(),
#     TotalPrecision(),
#     MeanPrecision2(),
#     MeankendallTau(),
#     MeanKendallTauAtK(2),
#     MeanKendallTauAtK(10),
#     MeanSpearmanR(),
#     MeanSpearmanRAtK(3),
# ]
#
# for metric in ltr_metrics:
#     print(metric, metric(y, y_pred, gids))
