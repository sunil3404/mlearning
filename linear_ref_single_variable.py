from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd



class Linear():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_train_test_data(self, x, y):
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        return train_x, test_x, train_y, test_y

    def train_data(self, train_x, train_y):
        model = LinearRegression()
        model.fit(train_x, train_y)
        return model 
    
    def test_data(self, test_x, test_y, model):
        predicted_price = []
        for row in test_x:
            price = model.predict([row])
            predicted_price.append(price)
        return predicted_price
    
    def test_data_linear(self, test_x, test_y, model):
        y_pred = model.predict(test_x)
        return y_pred

if __name__ == "__main__":
    boston_data = load_boston()
    b_data = boston_data.data
    target = boston_data.target
    column = boston_data.feature_names
    
    df = pd.DataFrame(np.array(b_data), columns=column)
    x = np.array(df)
    y = target

    linear = Linear(x, y)
    scaled_x = StandardScaler()
    x = scaled_x.fit_transform(x)
    train_x, test_x, train_y, test_y = linear.get_train_test_data(linear.x, linear.y)
    model  = linear.train_data(train_x, train_y)
    predicted_price  = linear.test_data(test_x, test_y, model)
    predicted_price_linear  = linear.test_data_linear(test_x, test_y, model)
    Actual_price = test_y

    boston_house_price  = pd.DataFrame(np.array(predicted_price), columns=['predicted_price'])
    boston_house_price['actual_price'] = np.array(test_y)

    print(boston_house_price.head())

    boston_house_price['error'] = boston_house_price['predicted_price'] - boston_house_price['actual_price']
    print(boston_house_price.head(10))
