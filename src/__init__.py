import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(PATH):
    df = pd.read_csv(PATH)
    return df.to_dict(orient='list')