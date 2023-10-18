import models
from quickdraw import QuickDrawDataGroup, QuickDrawData
import random
import numpy as np



quickdraw_model = models.get_lstm_model_quickdraw()
models.train_categorical()