import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from models.XGBoost import XGBoostModel


def lin_regression(train_data, train_true, test_data, test_true):
    train_std = StandardScaler().fit_transform(train_data)
    test_std = StandardScaler().fit_transform(test_data)

    clf = SGDRegressor(loss="squared_loss", penalty="l2")
    values = [10 ** -14, 10 ** -12, 10 ** -10, 10 ** -8, 10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0, 10 ** 2, 10 ** 4,
              10 ** 6]
    hyper_parameter = {"alpha": values}
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring="neg_mean_absolute_error", cv=3)
    best_parameter.fit(train_std, train_true)
    alpha = best_parameter.best_params_["alpha"]

    clf = SGDRegressor(loss="squared_loss", penalty="l2", alpha=alpha)
    clf.fit(train_std, train_true)
    train_pred = clf.predict(train_std)
    train_MAPE = mean_absolute_error(train_true, train_pred) / (sum(train_true) / len(train_true))
    train_MSE = mean_squared_error(train_true, train_pred)
    test_pred = clf.predict(test_std)
    test_MAPE = mean_absolute_error(test_true, test_pred) / (sum(test_true) / len(test_true))
    test_MSE = mean_squared_error(test_true, test_pred)
    joblib.dump(clf, "../static/lr-trained")

    return train_MAPE, train_MSE, test_MAPE, test_MSE


def randomForest(train_data, train_true, test_data, test_true):
    values = [10, 40, 80, 150, 600, 800]
    clf = RandomForestRegressor(n_jobs=-1)
    hyper_parameter = {"n_estimators": values}
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring="neg_mean_absolute_error", cv=3)
    best_parameter.fit(train_data, train_true)
    estimators = best_parameter.best_params_["n_estimators"]

    clf = RandomForestRegressor(n_estimators=estimators, n_jobs=-1)
    clf.fit(train_data, train_true)
    train_pred = clf.predict(train_data)
    train_MAPE = mean_absolute_error(train_true, train_pred) / (sum(train_true) / len(train_true))
    train_MSE = mean_squared_error(train_true, train_pred)
    test_pred = clf.predict(test_data)
    test_MAPE = mean_absolute_error(test_true, test_pred) / (sum(test_true) / len(test_true))
    test_MSE = mean_squared_error(test_true, test_pred)
    joblib.dump(clf, "../static/rf-trained")

    return train_MAPE, train_MSE, test_MAPE, test_MSE


def xgboost_reg(train_X, train_y, test_X, test_y, xg):
    test_pred = xg.cv(train_X, train_y).train().predict(test_X)
    test_MAPE = mean_absolute_error(test_y, test_pred) / (sum(test_y) / len(test_y))
    test_MSE = mean_squared_error(test_y, test_pred)
    print("xgboost_reg() - test MAPE: ", test_MAPE)
    print("xgboost_reg() - test MSE: ", test_MSE)
