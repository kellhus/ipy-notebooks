import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    return pd.DataFrame(np.load('data/' + filename)[()])
