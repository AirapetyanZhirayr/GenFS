from importlib import reload

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)

import pickle5 as pickle
import json
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import anytree

from tqdm import tqdm

import east

import re
import datefinder
from datetime import datetime, timedelta
import html

def find_dates(y):
    return list(datefinder.find_dates(y, base_date=datetime(1, 1, 1)))

def filter_dates(y):
    return [x for x in y if x > datetime(1900, 1, 1) and x <= datetime(2021, 4, 1)]



def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

