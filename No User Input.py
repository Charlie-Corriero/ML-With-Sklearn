import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    # idea, have the user pick which predictors they want to use and what they want to predict and do it using those
    # predictors

    # Loading the dataset
    data = pd.read_csv('dc_weather.csv')

    # Data cleaning
    data['severerisk'] = data['severerisk'].fillna(0)
    data['preciptype'] = data['preciptype'].fillna('none')

    # Encoding conditions to be used as a feature
    conditions_encoded = pd.get_dummies(data['conditions'], drop_first=True)

    # Selecting additional columns that I want to use as feature
    additional_columns = data[['temp', 'dew', 'humidity', 'windspeed', 'uvindex']]

    # Combining the encoded data and the additional columns
    combined_data = pd.concat([conditions_encoded.reset_index(drop=True), additional_columns.reset_index(drop=True)],
                              axis=1)

    # Getting features (x) and target (y)
    x = combined_data
    y = data['feelslike']

    # Asserting that the shape of x and y are the same
    assert x.shape[0] == y.shape[0], "The number of samples in X and y must be the same!"

    # Splitting the data into target and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initializing the model
    model = LinearRegression()

    # Training the model
    model.fit(x_train, y_train)

    # Making predictions
    predictions = model.predict(x_test)

    # Getting metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    n = len(y_test)
    p = x_test.shape[1]
    adj_r2 = adjusted_r2(r2, n, p)

    print('MSE: ', mse)
    print('R2: ', r2)
    print("Adjusted R2: ", adj_r2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
