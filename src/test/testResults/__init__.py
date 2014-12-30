
from src.commons.utils import utils

import operator

def _bestEstimatorsSortKey():
    return operator.itemgetter('startIndex', 'fold')


def loadResults():
    results = utils.load('/Users/noampeled/Dropbox/postDocMoshe/MEG/AnalyzeMEG/results.pkl')
    results = sorted(results, key=_bestEstimatorsSortKey())


if __name__ == '__main__':
    loadResults()