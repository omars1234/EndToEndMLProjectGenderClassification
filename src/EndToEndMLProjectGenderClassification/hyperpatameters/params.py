from sklearn.ensemble import (
    RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
)

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.linear_model import (
    LinearRegression,ElasticNet
)

from sklearn.neighbors import KNeighborsClassifier




models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier()
}


