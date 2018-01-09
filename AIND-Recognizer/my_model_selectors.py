import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        '''
        This is close, but not correct.

        N is the number of data points, f is the number of features:

        N, f = self.X.shape

        Having m as the num_components, The free parameters p are a sum of:

        The free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so m*(m-1)
        The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so m-1
        The number of means, which is m*f
        Number of covariances which is the size of the covars matrix, which for "diag" is m*f
        All of the above is equal to:

        p = m^2 +2mf-1
        '''
        best_n_components = 2
        best_BIC = float('inf')
        try:
            for i in range(self.min_n_components, self.max_n_components + 1):
                model_temp = self.base_model(i)
                logL_temp = model_temp.score(self.X, self.lengths)
                #
                num_param = i**2 + 2 * i * self.X.shape[1] - 1
                #
                BIC_temp = -2 * logL_temp + num_param * np.log(self.X.shape[0])
                if BIC_temp < best_BIC:
                    best_n_components = i
                    best_BIC = BIC_temp
            return self.base_model(best_n_components)
        except:
            return self.base_model(best_n_components)

        
     





class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_n_components = 2
        best_DIC = -float('inf')
        try:
            for i in range(self.min_n_components, self.max_n_components + 1):
                model_temp = self.base_model(i)
                logL_thisword = model_temp.score(self.X, self.lengths)
                logL_otherword_sum = 0
                otherword_num = 0
                for otherword, ddmymodel in self.words.items():
                    #print(otherword)
                    if otherword != self.this_word:
                        X_otherword, lengths_otherword = self.hwords[otherword]
                        logL_otherword = model_temp.score(X_otherword, lengths_otherword)
                        logL_otherword_sum += logL_otherword
                        otherword_num += 1
                #print(otherword_num)
                DIC_temp = logL_thisword - logL_otherword_sum/otherword_num
                if DIC_temp > best_DIC:
                    best_n_components = i
                    best_DIC = DIC_temp
            return self.base_model(best_n_components)
        except:
            return self.base_model(3)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        split_method = KFold()
        best_avg_logL = -float('inf')
        
        best_n_components = 2
        #try:
        try:
            for i in range(self.min_n_components, self.max_n_components + 1):
                if len(self.sequences) > 2:
                    avg_logL = 0
                    num_fold = 0
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        num_fold = num_fold + 1
                        #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        logL_thisfold = hmm_model.score(X_test, lengths_test)
                        avg_logL = avg_logL + logL_thisfold
                    avg_logL = avg_logL/num_fold
                    #print(avg_logL)
                    #print(num_fold)
                    if avg_logL > best_avg_logL:
                        best_n_components = i
                        best_avg_logL = avg_logL
                else:
                    best_n_components = 3
        except:
            best_n_components = 3
        return self.base_model(best_n_components)
        #except:
        #    return self.base_model(best_n_components)    


                





