# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Threshold Optimization Post Processing algorithm.

This is based on M. Hardt, E. Price, N. Srebro's paper
"`Equality of Opportunity in Supervised Learning
<https://arxiv.org/pdf/1610.02413.pdf>`_" for binary
classification with one categorical sensitive feature [1]_.

References
----------
.. [1] M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in
   Supervised Learning," arXiv.org, 07-Oct-2016. [Online]. Available:
   https://arxiv.org/abs/1610.02413.

"""

import logging
from warnings import warn

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted


from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.utils._common import _get_soft_predictions
from fairlearn.utils._input_validation import _KW_CONTROL_FEATURES, _validate_and_reformat_input



# various error messages
DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "{} need to be of equal length."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."
MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE = (
    "Post processing currently only supports a single column in {}."
)

SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE = (
    "The provided scores data contains multiple columns."
)
UNEXPECTED_DATA_TYPE_ERROR_MESSAGE = "Unexpected data type {} encountered."

logger = logging.getLogger(__name__)


# Simple constraints are described by metrics with values between 0 and 1,
# which attain both extremes as the threshold goes from -Inf to Inf.
# These metrics are also required to be "moments" in the same sense as
# required by fairlearn.reductions, so that the interpolation is possible.
SIMPLE_CONSTRAINTS = {
    "selection_rate_parity": "selection_rate",
    "demographic_parity": "selection_rate",
    "false_positive_rate_parity": "false_positive_rate",
    "false_negative_rate_parity": "false_negative_rate",
    "true_positive_rate_parity": "true_positive_rate",
    "true_negative_rate_parity": "true_negative_rate",
}

ALL_CONSTRAINTS = list(SIMPLE_CONSTRAINTS.keys()) + ["equalized_odds"]

# Any "moment" is allowed as a performance metric for simple constraints.
OBJECTIVES_FOR_SIMPLE_CONSTRAINTS = {
    "selection_rate",
    "true_positive_rate",
    "true_negative_rate",
    "accuracy_score",
    "balanced_accuracy_score",
}

# Besides simple constraints we also allow 'equalized_odds' as a constraint.

# For equalized odds, we only allow objectives that are non-decreasing in true_positives,
# when holding n, positives, negatives, true_negatives, and false_positives fixed.
OBJECTIVES_FOR_EQUALIZED_ODDS = {
    "accuracy_score",
    "balanced_accuracy_score",
}

NO_CONTROL_FEATURES = "Control features are not supported by ThresholdOptimizer"
NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE = (
    "Currently only the following constraints are supported: {}.".format(
        ", ".join(sorted(ALL_CONSTRAINTS))
    )
)
NOT_SUPPORTED_OBJECTIVES_FOR_SIMPLE_CONSTRAINTS_ERROR_MESSAGE = (
    "For {{}} only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_SIMPLE_CONSTRAINTS))
    )
)
NOT_SUPPORTED_OBJECTIVES_FOR_EQUALIZED_ODDS_ERROR_MESSAGE = (
    "For equalized_odds only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_EQUALIZED_ODDS))
    )
)


class ThresholdOptimizerNew(ThresholdOptimizer):
    """A classifier based on the threshold optimization approach.

    The classifier is obtained by applying group-specific thresholds to the
    provided estimator. The thresholds are chosen to optimize the provided
    performance objective subject to the provided fairness constraints.

    Read more in the :ref:`User Guide <postprocessing>`.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator <https://scikit-learn.org/stable/developers/develop.html#estimators>`_  # noqa
        whose output is postprocessed.

    constraints : str, default='demographic_parity'
        Fairness constraints under which threshold optimization is performed.
        Possible inputs are:

            'demographic_parity', 'selection_rate_parity' (synonymous)
                match the selection rate across groups

            '{false,true}_{positive,negative}_rate_parity'
                match the named metric across groups

            'equalized_odds'
                match true positive and false positive rates across groups

    objective : str, default='accuracy_score'
        Performance objective under which threshold optimization is performed.
        Not all objectives are allowed for all types of constraints.
        Possible inputs are:

            'accuracy_score', 'balanced_accuracy_score'
                allowed for all constraint types

            'selection_rate', 'true_positive_rate', 'true_negative_rate',
                allowed for all constraint types except 'equalized_odds'

    grid_size : int, default=1000
        The values of the constraint metric are discretized according to the
        grid of the specified size over the interval [0,1] and the optimization
        is performed with respect to the constraints achieving those values. In
        case of 'equalized_odds' the constraint metric is the false positive
        rate.

    flip : bool, default=False
        If True, then allow flipping the decision if it improves the resulting

    prefit : bool, default=False
        If True, avoid refitting the given estimator. Note that when used with
        :func:`sklearn.model_selection.cross_val_score`,
        :class:`sklearn.model_selection.GridSearchCV`, this will result in an
        error. In that case, please use ``prefit=False``.

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

        - 'auto': use one of ``predict_proba``, ``decision_function``, or
          ``predict``, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': use the hard values reported by the `predict` method if
          estimator is a classifier, and the regression values if estimator is
          a regressor. This is equivalent to what is done in [1]_.

        .. versionadded:: 0.7
            In previous versions only the ``predict`` method was used
            implicitly.

        .. versionchanged:: 0.7
            From version 0.7, 'predict' is deprecated as the default value and
            the default will change to 'auto' from v0.10.

    Notes
    -----
    The procedure is based on the algorithm of
    `Hardt et al. (2016) <https://arxiv.org/abs/1610.02413>`_ [1]_.

    References
    ----------
    .. [1] M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in
       Supervised Learning," arXiv.org, 07-Oct-2016.
       [Online]. Available: https://arxiv.org/abs/1610.02413.

    Examples
    --------
    >>> from fairlearn.postprocessing import ThresholdOptimizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> X                  = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    >>> y                  = [ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  1 ,  0 ,  0 ,  0 ]
    >>> sensitive_features = ["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"]
    >>> unmitigated_lr = LogisticRegression().fit(X, y)
    >>> postprocess_est = ThresholdOptimizer(
    ...                    estimator=unmitigated_lr,
    ...                    constraints="false_negative_rate_parity",
    ...                    objective="balanced_accuracy_score",
    ...                    prefit=True,
    ...                    predict_method='predict_proba')
    >>> postprocess_est.fit(X, y, sensitive_features=sensitive_features)
    ThresholdOptimizer(constraints='false_negative_rate_parity',
                       estimator=LogisticRegression(),
                       objective='balanced_accuracy_score',
                       predict_method='predict_proba', prefit=True)
    """

    def __init__(
        self,
        *,
        estimator=None,
        constraints="demographic_parity",
        objective="accuracy_score",
        grid_size=1000,
        flip=False,
        prefit=False,
        predict_method="deprecated",
    ):
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.grid_size = grid_size
        self.flip = flip
        self.prefit = prefit
        self.predict_method = predict_method

    def fit(self, X, y, *, sensitive_features, **kwargs):
        """Fit the model.

        The fit is based on training features and labels, sensitive features,
        as well as the fairness-unaware predictor or estimator. If an estimator was passed
        in the constructor this fit method will call `fit(X, y, **kwargs)` on said estimator.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The label vector
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, or pandas.Series
            sensitive features to identify groups by
        """
        if self.estimator is None:
            raise ValueError('No estimator')

        if self.constraints in SIMPLE_CONSTRAINTS:
            if self.objective not in OBJECTIVES_FOR_SIMPLE_CONSTRAINTS:
                raise ValueError(
                    NOT_SUPPORTED_OBJECTIVES_FOR_SIMPLE_CONSTRAINTS_ERROR_MESSAGE.format(
                        self.constraints
                    )
                )
        elif self.constraints == "equalized_odds":
            if self.objective not in OBJECTIVES_FOR_EQUALIZED_ODDS:
                raise ValueError(
                    NOT_SUPPORTED_OBJECTIVES_FOR_EQUALIZED_ODDS_ERROR_MESSAGE
                )
        else:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        if self.predict_method == "deprecated":
            warn(
                "'predict_method' default value is changed from 'predict' to "
                "'auto'. Explicitly pass `predict_method='predict' to "
                "replicate the old behavior, or pass `predict_method='auto' "
                "or other valid values to silence this warning.",
                FutureWarning,
            )
            self._predict_method = "predict"
        else:
            self._predict_method = self.predict_method

        if kwargs.get(_KW_CONTROL_FEATURES) is not None:
            raise ValueError(NO_CONTROL_FEATURES)

        _, _, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            enforce_binary_labels=True,
        )

        # postprocessing can't handle 0/1 as floating point numbers, so this
        # converts it to int
        if type(y) in [np.ndarray, pd.DataFrame, pd.Series]:
            y = y.astype(int)
        else:
            y = [int(y_val) for y_val in y]



        scores = self.estimator
        if self.constraints == "equalized_odds":
            self.x_metric_ = "false_positive_rate"
            self.y_metric_ = "true_positive_rate"
            threshold_optimization_method = (
                self._threshold_optimization_for_equalized_odds
            )
        else:
            self.x_metric_ = SIMPLE_CONSTRAINTS[self.constraints]
            self.y_metric_ = self.objective
            threshold_optimization_method = (
                self._threshold_optimization_for_simple_constraints
            )

        self.interpolated_thresholder_ = threshold_optimization_method(
            sensitive_feature_vector, y, scores
        )
        return self

    def predict(self, X, *, sensitive_features, random_state=None):
        """Predict label for each sample in X while taking into account \
            sensitive features.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, pandas.Series
            sensitive features to identify groups by
        random_state : int or :class:`numpy.random.RandomState` instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        numpy.ndarray
            The prediction in the form of a scalar or vector.
            If `X` represents the data for a single example the result will be
            a scalar. Otherwise the result will be a vector.
        """
        check_is_fitted(self)
        return self.interpolated_thresholder_.predict(
            X, sensitive_features=sensitive_features, random_state=random_state
        )