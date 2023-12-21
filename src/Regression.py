#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:29:04 2023

@author: juan
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

train = pd.read_csv("../data/train.csv", na_values="NaN") # Definimos na_values para identificar bien los valores perdidos
print(train.columns)