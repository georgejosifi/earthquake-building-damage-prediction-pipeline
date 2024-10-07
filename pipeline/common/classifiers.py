from enum import Enum

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
import mord



class ClassificationAlgorithm(str, Enum):
    AdaBoostClassifier = "AdaBoostClassifier"
    DecisionTreeClassifier = "DecisionTreeClassifier"
    KNeighborsClassifier = "KNeighborsClassifier"
    RadiusNeighborsClassifier = "RadiusNeighborsClassifier"
    LinearSVC = "LinearSVC",
    NuSVC = "NuSVC",
    SVC = "SVC"
    LogisticRegression = "LogisticRegression"
    MLPClassifier = "MLPClassifier"
    PassiveAggressiveClassifier = "PassiveAggressiveClassifier"
    RandomForestClassifier = "RandomForestClassifier"
    RidgeClassifier = "RidgeClassifier"
    SGDClassifier = "SGDClassifier"
    Perceptron = "Perceptron"
    GaussianNB = "GaussianNB"
    # XGBoostClassifier = "XGBoostClassifier"
    LogisticAT = "LogisticAT"
    LogisticIT = "LogisticIT"
    LogisticSE = "LogisticSE"


classifiers = {
    ClassificationAlgorithm.AdaBoostClassifier: AdaBoostClassifier,
    ClassificationAlgorithm.DecisionTreeClassifier: DecisionTreeClassifier,
    ClassificationAlgorithm.KNeighborsClassifier: KNeighborsClassifier,
    ClassificationAlgorithm.RadiusNeighborsClassifier: RadiusNeighborsClassifier,
    ClassificationAlgorithm.LinearSVC: LinearSVC,
    ClassificationAlgorithm.NuSVC: NuSVC,
    ClassificationAlgorithm.SVC: SVC,
    ClassificationAlgorithm.LogisticRegression: LogisticRegression,
    ClassificationAlgorithm.MLPClassifier: MLPClassifier,
    ClassificationAlgorithm.PassiveAggressiveClassifier: PassiveAggressiveClassifier,
    ClassificationAlgorithm.RandomForestClassifier: RandomForestClassifier,
    ClassificationAlgorithm.RidgeClassifier: RidgeClassifier,
    ClassificationAlgorithm.SGDClassifier: SGDClassifier,
    ClassificationAlgorithm.Perceptron: Perceptron,
    ClassificationAlgorithm.GaussianNB: GaussianNB,
    # ClassificationAlgorithm.XGBoostClassifier: XGBClassifier
    ClassificationAlgorithm.LogisticAT: mord.LogisticAT,
    ClassificationAlgorithm.LogisticIT: mord.LogisticIT,
    ClassificationAlgorithm.LogisticSE: mord.LogisticSE,
}


def get_classification_algorithm(classifier_type: str, classifier_parameters):
    try:
        classifier = classifiers[ClassificationAlgorithm[classifier_type]]
    except:
        print(
            f"Error: Classifier type must be one of the following: {[e.value for e in ClassificationAlgorithm]}, but {classifier_type} was given!")
        exit()
    
    instance = classifier(**classifier_parameters)
    return instance