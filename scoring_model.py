import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import os
from sklearn import svm
import pickle

class ScoringModel():
    def __init__(self, data_path, label,  model_path, categories):
        self.data = data_path
        self.label = label
        self.categories = categories
        self.model_path = model_path
        self.X = self.data.drop(labels=[self.label], axis=1).values
        self.y = self.data[self.label].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42, stratify=self.y)
        if os.path.exists(self.model_path):
            print("The model is ready")
            self.load_model = pickle.load(open(self.model_path, "rb"))
        else:
            print("The model does not exist")
            self.train_model()


    def make_meshgrid(self, x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out


    def train_model(self):
        if os.path.exists(self.model_path):
            print("The model is ready")
        else:
            linearSvcModel = svm.LinearSVC(C=1, max_iter=10000)
            linearSvcModel.fit(self.X_train, self.y_train)
            filename = self.model_path

            pickle.dump(linearSvcModel, open(filename, 'wb'))

    def compute_score(self, candidate):
        clf = CalibratedClassifierCV(self.load_model)
        clf = clf.fit(self.X_train, self.y_train)

        output = clf.predict_proba(candidate)
        d = {}
        d[self.categories[0]] = str(round(output[0][0]*100,2))+" %"
        d[self.categories[1]] = str(round(output[0][1]*100,2))+" %"
        d[self.categories[2]] = str(round(output[0][2]*100,2))+" %"
        return d


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    df_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

    scoring_model = ScoringModel(df_data, 'Species', 'model.sav', ['0','1','2'])
    print(scoring_model.data)
    print("X:", [[7.3, 2.9, 6.3, 1.8]])

    score = scoring_model.compute_score([[7.3, 2.9, 6.3, 1.8]])

    print(score)