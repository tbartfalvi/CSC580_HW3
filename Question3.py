# char_nb_languageid.py
import os
import math
import numpy as np
from collections import Counter, defaultdict
from math import log, exp
from scipy.special import gammaln  # for log factorial via gammaln(n+1)

language_id_files = 'data/languageID'
# I could not figure this question out in python.  I attempted to use ChatGPT, 
# but I still could not do it correctly.