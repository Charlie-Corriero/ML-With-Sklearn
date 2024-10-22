import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    # TODO: Allow the user to input their own file somehow

    # Loading the dataset
    data = pd.read_csv('dc_weather.csv')

    # Data cleaning
    data['severerisk'] = data['severerisk'].fillna(0)
    data['preciptype'] = data['preciptype'].fillna('none')
    data = data.drop(columns=['name', 'datetime', 'sunrise', 'sunset', 'description', 'icon', 'stations'])
    fields = []
    for column in data.columns:
        fields.append(column)

    # Getting the target from the user
    print("Data Fields\n_________________________")
    for column in data.columns:
        print(column)
    print("_______________________________")
    y_choice = ""
    while y_choice not in fields:
        y_choice = input("Please enter a field from above that you would like to predict (the target): ")

    y = data[y_choice]

    # Removing the target column from the choices of features
    data = data.drop(columns=[y_choice])
    fields.remove(y_choice)

    # Getting the features from the user
    x_list = []
    print("_______________________________")
    while True:
        x_choice = input(
            "Please enter a field from above that you would like to use as a predictor (feature) or type 'done' to "
            "finish: ")

        # Check if the user wants to finish
        if x_choice.lower() == "done":
            break

        # Check if the choice is valid
        if x_choice in fields:
            x_list.append(x_choice)
            print("Successfully added ", x_choice, " as a feature")
        else:
            print("Invalid choice. Please enter a valid field from the list.")

    # Encoding categorical variables if there are any, and adding features to their variable
    features = data[x_list]

    # Identify categorical columns in needed
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

    # Encode the columns and concatenate if there are categorical columns
    if categorical_cols:
        features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    else:
        features_encoded = features.copy()

    # Asserting that the shape of x and y are the same
    assert features_encoded.shape[0] == y.shape[0], "The number of samples in X and y must be the same!"

    # Splitting the data into target and training
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, y, test_size=0.2, random_state=42)
    print("_______________________________")
    # Letting the user pick the model
    model_choice = ""
    model_list = ['linear', 'ridge', 'lasso', 'elastic net']
    while model_choice not in model_list:
        model_choice = input("Please enter the regression model you would like to use: ")

    if model_choice.lower() == "linear":
        model = LinearRegression()
    elif model_choice.lower() == "ridge":
        model = Ridge()
    elif model_choice.lower() == "lasso":
        model = Lasso()
    elif model_choice.lower() == "elastic net":
        model = ElasticNet()

    # Training the data
    model.fit(X_train, y_train)

    # Making predictions
    predictions = model.predict(X_test)

    # Getting metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = adjusted_r2(r2, n, p)
    print("Metrics\n_______________________")
    print('MSE: ', mse)
    print('R2: ', r2)
    print("Adjusted R2: ", adj_r2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
