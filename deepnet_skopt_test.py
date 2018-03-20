from deepnet.deep_model import DeepKDModel
from preprocess import load_data
x_train, x_test, y_train, y_test = load_data.load(one_hot=True, fill_mode='mean')
model = DeepKDModel(verbose=True)
model.optimize_hyperparameters(x_train, x_test, y_train, y_test)