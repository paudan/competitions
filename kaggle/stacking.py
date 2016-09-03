# -*- coding: utf-8 -*-
import numpy as np
from itertools import izip
import scipy.sparse as sps
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble.base import BaseEnsemble
from sklearn.utils.validation import assert_all_finite

class Stacking(BaseEnsemble):
    """Implements stacking/blending.

    Parameters
    ----------
    meta_estimator : string or callable
        May be one of "best", "vote", "average", or any classifier or 
        regressor constructor

    estimators : iterator
        An iterable of estimators; each must support predict_proba()

    cv : iterator
        A cross validation object. Base (level-0) estimators are trained on 
        the training folds, then the meta (level-1) estimator is trained on 
        the testing folds.

    stackingc : bool
        Whether to use StackingC or not. For more information, refer to the 
        following paper:

        Reference:
          A. K. Seewald, "How to Make Stacking Better and Faster While Also 
          Taking Care of an Unknown Weakness," 2002.

    kwargs :
        Arguments passed to instantiate meta_estimator.

    References
    ----------
    .. [1] D. H. Wolpert, "Stacked Generalization", 1992.

    """

    # TODO: Support different features for each estimator.
    # TODO: Support "best", "vote", and "average" for already trained model.
    # TODO: Allow saving of estimators, so they need not be retrained when 
    #       trying new stacking methods.

    def __init__(self, meta_estimator, estimators,
                 cv, stackingc=True, proba=True,
                 **kwargs):
        self.estimators_ = estimators
        self.n_estimators_ = len(estimators)
        self.cv_ = cv
        self.stackingc_ = stackingc
        self.proba_ = proba

        if stackingc:
            if isinstance(meta_estimator, str) or not issubclass(meta_estimator, RegressorMixin):
                raise Exception('StackingC only works with a regressor.')

        if isinstance(meta_estimator, str):
            if meta_estimator not in ('best',
                                      'average',
                                      'vote'):
                raise Exception('Invalid meta estimator: {0}'.format(meta_estimator))
            raise Exception('"{0}" meta estimator not implemented'.format(meta_estimator))
        elif issubclass(meta_estimator, ClassifierMixin):
            self.meta_estimator_ = meta_estimator(**kwargs)
        elif issubclass(meta_estimator, RegressorMixin):
            self.meta_estimator_ = MRLR(meta_estimator, stackingc, **kwargs)
        else:
            raise Exception('Invalid meta estimator: {0}'.format(meta_estimator))

    def _base_estimator_predict(self, e, X):
        """Predict label values with the specified estimator on predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the specified 
            estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
        pred = e.predict(X)
        assert_all_finite(pred)
        return pred

    def _base_estimator_predict_proba(self, e, X):
        """Predict label probabilities with the specified estimator on 
        predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the specified 
            estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
        pred = e.predict_proba(X)
        assert_all_finite(pred)
        return pred

    def _make_meta(self, X):
        """Make the feature set for the meta (level-1) estimator.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data.

        Returns
        -------
        An n x len(self.estimators_) array of meta-level features.
        """
        rows = []
        for e in self.estimators_:
            if self.proba_:
                # Predict label probabilities
                pred = self._base_estimator_predict_proba(e, X)
            else:
                # Predict label values
                pred = self._base_estimator_predict(e, X)
            rows.append(pred)
        return np.hstack(rows)

    def fit(self, X, y):
        """Fit the estimator given predictor(s) X and target y.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data on which to fit.

        y : array of shape = [n_samples]
            The actual outputs (class data).
        """
        # Build meta data.
        X_meta = [] # meta-level features
        y_meta = [] # meta-level labels

        print 'Training and validating the base (level-0) estimator(s)...'
        print
        for i, (a, b) in enumerate(self.cv_):
            print 'Fold [%s]' % (i)

            X_a, X_b = X[a], X[b] # training and validation features
            y_a, y_b = y[a], y[b] # training and validation labels

            # Fit each base estimator using the training set for the fold.
            for j, e in enumerate(self.estimators_):
                print '  Training base (level-0) estimator %d...' % (j),
                if isinstance(e, KerasClassifier):
                    e.fit(X_a.toarray() if sps.issparse(X_a) else X_a, 
                          y_a.toarray() if sps.issparse(y_a) else y_a)
                else:
                     e.fit(X_a, y_a)
                print 'done.'

            proba = self._make_meta(X_b)
            X_meta.append(proba)
            y_meta.append(y_b)
        print

        X_meta = np.vstack(X_meta)
        if y_meta[0].ndim == 1:
            y_meta = np.hstack(y_meta)
        else:
            y_meta = np.vstack(y_meta)

        # Train meta estimator.
        print 'Training meta (level-1) estimator...',
        self.meta_estimator_.fit(X_meta, y_meta)
        print 'done.'

        # Re-train base estimators on full data.
        for j, e in enumerate(self.estimators_):
            print 'Re-training base (level-0) estimator %d on full data...' % (j),
            e.fit(X, y)
            print 'done.'

    def predict(self, X):
        """Predict label values with the fitted estimator on predictor(s) X.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted output.

        Returns
        -------
        array of shape = [n_samples]
            The predicted label values of the input samples.
        """
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict(X_meta)

    def predict_proba(self, X):
        """Predict label probabilities with the fitted estimator on 
        predictor(s) X.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted output.

        Returns
        -------
        array of shape = [n_samples]
            The predicted label probabilities of the input samples.
        """
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict_proba(X_meta)