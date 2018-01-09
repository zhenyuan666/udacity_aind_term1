import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    for i in range(test_set.num_items):
        myX, mylengths = test_set.get_item_Xlengths(i)
        prob_dict = {}
        max_logL = -99999999
        max_logL_word = None
        for word, mymodel in models.items():
            try:
                mylogL = mymodel.score(myX, mylengths)
                prob_dict[word] = mylogL
                if mylogL > max_logL:
                    max_logL = mylogL
                    max_logL_word = word
            except:
                prob_dict[word] = -99999999
        probabilities.append(prob_dict)
        guesses.append(max_logL_word)
    return probabilities, guesses






