# char_nb_languageid.py
import os
import math
import numpy as np
from collections import Counter, defaultdict
from math import log, exp
from scipy.special import gammaln  # for log factorial via gammaln(n+1)

language_id_files = 'data/languageID'