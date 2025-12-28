import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

class LinearRegr:
    def fit(self, X, Y):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta minimalizujace kwadratowa funkcje kosztu L uzywajac wzoru.
        # Uwaga: przed zastosowaniem wzoru do X nalezy dopisac kolumne zlozona z jedynek.
        n, m = X.shape
        self.theta = np.zeros((m+1))
        X1 = np.c_[np.ones(X.shape[0]), X]
        X1_T = X1.T
        self.theta = np.linalg.inv(X1_T @ X1) @ X1_T @ Y
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (k, m)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, m = X.shape
        X1 = np.c_[np.ones(X.shape[0]), X]
        Y1 = X1 @ self.theta
        return Y1


def test_RegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,2,10]).reshape((3,1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("test one dim:")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected))

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("test three dim")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected))
    
test_RegressionInOneDim()
test_RegressionInThreeDim()