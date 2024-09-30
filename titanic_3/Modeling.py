import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

class MyModelingProcess:
    def __init__(self, _X, _y):
        self.X = copy.copy(_X.values)
        self.y = copy.copy(_y.values)
        self.models = []
        self.n_features = _X.shape[1]

    def perceptron_grid(self):
        pipe_pca = Pipeline([
            ('classifier', Perceptron())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'classifier__penalty': [None, 'l2', 'l1', 'elasticnet'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__max_iter': [1000, 1500, 2000],
            'classifier__tol': [1e-3],
        }

        # 그리드 서치 객체 생성
        perceptron_grid = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        perceptron_grid.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('perceptron', perceptron_grid.best_estimator_, perceptron_grid.best_score_))
        self.save_model('perceptron', ('perceptron', perceptron_grid.best_estimator_, perceptron_grid.best_score_))
        # 최적의 파라미터와 점수 출력
        print("PCA grid search result: ")
        print(perceptron_grid.best_params_)
        print(perceptron_grid.best_score_)
        print()

    def perceptron_pca(self):
        pipe_pca = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('classifier', Perceptron())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__penalty': [None, 'l2', 'l1', 'elasticnet'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__max_iter': [1000],
            'classifier__tol': [1e-3],
        }

        # 그리드 서치 객체 생성
        perceptron_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        perceptron_grid_pca.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('perceptron_pca', perceptron_grid_pca.best_estimator_, perceptron_grid_pca.best_score_))
        self.save_model('perceptron_pca', ('perceptron_pca', perceptron_grid_pca.best_estimator_, perceptron_grid_pca.best_score_))
        # 최적의 파라미터와 점수 출력
        print("Best parameters with PCA:")
        print(perceptron_grid_pca.best_params_)
        print("Best cross-validation accuracy with PCA:")
        print(perceptron_grid_pca.best_score_)

    def perceptron_sbs(self):
        # 퍼셉트론 모델 정의
        perceptron = Perceptron(max_iter=1000, tol=1e-3)

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(perceptron, n_features_to_select=8, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', Perceptron())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__penalty': [None, 'l2', 'l1', 'elasticnet'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__max_iter': [1000],
            'classifier__tol': [1e-3],
        }

        # 그리드 서치 객체 생성
        perceptron_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        perceptron_grid_sbs.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('perceptron_sbs', perceptron_grid_sbs.best_estimator_, perceptron_grid_sbs.best_score_))
        self.save_model('perceptron_sbs', ('perceptron_sbs', perceptron_grid_sbs.best_estimator_, perceptron_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(perceptron_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(perceptron_grid_sbs.best_score_)

    def LogisticRegression(self):
        # 파이프라인 구성
        pipe = Pipeline([
            ('classifier', LogisticRegression())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga', 'liblinear'],
            'classifier__max_iter': [1000, 1500, 2000],
        }

        # 그리드 서치 객체 생성
        LogisticRegression_grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

        # 모델 학습
        LogisticRegression_grid.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('LogisticRegression', LogisticRegression_grid.best_estimator_, LogisticRegression_grid.best_score_))
        self.save_model('LogisticRegression', ('LogisticRegression', LogisticRegression_grid.best_estimator_, LogisticRegression_grid.best_score_))
        # 최적의 파라미터와 점수 출력
        print("LogisticRegression grid search result: ")
        print(LogisticRegression_grid.best_params_)
        print(LogisticRegression_grid.best_score_)
        print()

    def LogisticRegression_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', LogisticRegression())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1,self.n_features)],
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga', 'liblinear'],
            'classifier__max_iter': [1000, 2000],
        }

        # 그리드 서치 객체 생성
        LogisticRegression_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        LogisticRegression_grid_pca.fit(self.X, self.y)


        # 최적의 모델 저장
        self.models.append(('LogisticRegression_pca', LogisticRegression_grid_pca.best_estimator_, LogisticRegression_grid_pca.best_score_))
        self.save_model('LogisticRegression_pca',('LogisticRegression_pca', LogisticRegression_grid_pca.best_estimator_, LogisticRegression_grid_pca.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with PCA:")
        print(LogisticRegression_grid_pca.best_params_)
        print("Best cross-validation accuracy with PCA:")
        print(LogisticRegression_grid_pca.best_score_)

    def LogisticRegression_sbs(self):
        # 로지스틱 회귀 모델 정의
        logreg = LogisticRegression(max_iter=1000, solver='saga')

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(logreg, n_features_to_select=8, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', LogisticRegression())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga', 'elasticnet'],
        }

        # 그리드 서치 객체 생성
        LogisticRegression_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        LogisticRegression_grid_sbs.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('LogisticRegression_sbs', LogisticRegression_grid_sbs.best_estimator_, LogisticRegression_grid_sbs.best_score_))
        self.save_model('LogisticRegression_sbs',('LogisticRegression_sbs', LogisticRegression_grid_sbs.best_estimator_, LogisticRegression_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(LogisticRegression_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(LogisticRegression_grid_sbs.best_score_)

    def SVM(self):
        # 파이프라인 구성
        pipe = Pipeline([
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear'],
            'classifier__max_iter': [1000, 1500, 2000]
        }

        # 그리드 서치 객체 생성
        SVC_grid = GridSearchCV(pipe, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        SVC_grid.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('SVC_grid', SVC_grid.best_estimator_, SVC_grid.best_score_))
        self.save_model('SVC_grid',('SVC_grid',  SVC_grid.best_estimator_, SVC_grid.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SVN:")
        print(SVC_grid.best_params_)
        print(SVC_grid.best_score_)


    def SVM_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear'],
        }

        # 그리드 서치 객체 생성
        SVC_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        SVC_grid_pca.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('SVC_pca', SVC_grid_pca.best_estimator_, SVC_grid_pca.best_score_))
        self.save_model('SVC_grid_pca', ('SVC_pca', SVC_grid_pca.best_estimator_, SVC_grid_pca.best_score_))
        

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SVM_PCA:")
        print(SVC_grid_pca.best_params_)
        print(SVC_grid_pca.best_score_)

    def SVM_sbs(self):
        # SVM 모델 정의
        svm = SVC(kernel='linear')

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(svm, n_features_to_select=8, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear'],
        }

        # 그리드 서치 객체 생성
        SVC_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        SVC_grid_sbs.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('SVC_sbs', SVC_grid_sbs.best_estimator_, SVC_grid_sbs.best_score_))
        self.save_model('SVC_grid_sbs', ('SVC_sbs', SVC_grid_sbs.best_estimator_, SVC_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(SVC_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(SVC_grid_sbs.best_score_)

    def SVM_kernel(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
            'classifier__gamma': [0.001, 0.01, 0.1, 1],
        }

        # 그리드 서치 객체 생성
        SVM_kernel_grid = GridSearchCV(pipe_pca, param_grid, cv=5, scoring='accuracy')

        # 모델 학습,
        SVM_kernel_grid.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('SVM_kernel', SVM_kernel_grid.best_estimator_, SVM_kernel_grid.best_score_))
        self.save_model('SVC_kernel_grid', ('SVM_kernel', SVM_kernel_grid.best_estimator_, SVM_kernel_grid.best_score_))
        # 최적의 파라미터와 점수 출력
        print("SVM_kernel Best parameters:")
        print(SVM_kernel_grid.best_params_)
        print(SVM_kernel_grid.best_score_)

    def SVM_kernel_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
            'classifier__gamma': [0.001, 0.01, 0.1, 1],
        }

        # 그리드 서치 객체 생성
        SVM_kernel_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        SVM_kernel_grid_pca.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('SVM_kerne_pca', SVM_kernel_grid_pca.best_estimator_, SVM_kernel_grid_pca.best_score_))
        self.save_model('SVC_kernel_grid_pca', ('SVM_kerne_pca', SVM_kernel_grid_pca.best_estimator_, SVM_kernel_grid_pca.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with PCA:")
        print(SVM_kernel_grid_pca.best_params_)
        print("Best cross-validation accuracy with PCA:")
        print(SVM_kernel_grid_pca.best_score_)

    def SVM_kernel_sbs(self):
        # SVM 모델 정의
        svm = SVC()

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(svm, n_features_to_select=8, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', SVC())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
            'classifier__gamma': [0.001, 0.01, 0.1, 1],
        }

        # 그리드 서치 객체 생성
        SVM_kernel_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        SVM_kernel_grid_sbs.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('SVM_kernel_sbs', SVM_kernel_grid_sbs.best_estimator_, SVM_kernel_grid_sbs.best_score_))
        self.save_model('SVC_kernel_grid_sbs', ('SVM_kernel_sbs', SVM_kernel_grid_sbs.best_estimator_, SVM_kernel_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(SVM_kernel_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(SVM_kernel_grid_sbs.best_score_)
    

    def DecisionTree(self):
        # 파이프라인 구성
        pipe = Pipeline([
            ('classifier', DecisionTreeClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        DecisionTree_grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
        
        # 모델 학습
        DecisionTree_grid.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('DecisionTree', DecisionTree_grid.best_estimator_, DecisionTree_grid.best_score_))
        self.save_model('DecisionTree',('DecisionTree', DecisionTree_grid.best_estimator_, DecisionTree_grid.best_score_))

        # 최적의 파라미터와 점수 출력
        print("DecisionTree Best parameters:")
        print(DecisionTree_grid.best_params_)
        print(DecisionTree_grid.best_score_)

    def DecisionTree_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', DecisionTreeClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        DecisionTree_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')
        
        # 모델 학습
        DecisionTree_grid_pca.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('DecisionTree_pca', DecisionTree_grid_pca.best_estimator_, DecisionTree_grid_pca.best_score_))
        self.save_model('DecisionTree_pca', ('DecisionTree_pca', DecisionTree_grid_pca.best_estimator_, DecisionTree_grid_pca.best_score_))
 
        # 최적의 파라미터와 점수 출력
        print("Best parameters with PCA:")
        print(DecisionTree_grid_pca.best_params_)
        print("Best cross-validation accuracy with PCA:")
        print(DecisionTree_grid_pca.best_score_)

    def DecisionTree_sbs(self):
        # 결정 트리 모델 정의
        dtree = DecisionTreeClassifier()

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(dtree, n_features_to_select=10, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', DecisionTreeClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        DecisionTree_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        DecisionTree_grid_sbs.fit(self.X, self.y)
        self.models.append(DecisionTree_grid_sbs)

        # 최적의 모델 저장
        self.models.append(('DecisionTree_sbs', DecisionTree_grid_sbs.best_estimator_, DecisionTree_grid_sbs.best_score_))
        self.save_model('DecisionTree_sbs', ('DecisionTree_sbs', DecisionTree_grid_sbs.best_estimator_, DecisionTree_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(DecisionTree_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(DecisionTree_grid_sbs.best_score_)

    def RandomForest(self):
        # 파이프라인 구성
        pipe = Pipeline([
            ('classifier', RandomForestClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        RandomForest_grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # 모델 학습
        RandomForest_grid.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('RandomForest', RandomForest_grid.best_estimator_, RandomForest_grid.best_score_))
        self.save_model('RandomForest',('RandomForest', RandomForest_grid.best_estimator_, RandomForest_grid.best_score_))
        # 최적의 파라미터와 점수 출력
        print("Best parameters with RandomForest:")
        print(RandomForest_grid.best_params_)
        print(RandomForest_grid.best_score_)

    def RandomForest_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', RandomForestClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        RandomForest_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy', n_jobs=-1)

        # 모델 학습
        RandomForest_grid_pca.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('RandomForest_pca', RandomForest_grid_pca.best_estimator_, RandomForest_grid_pca.best_score_))
        self.save_model('RandomForest_pca',('RandomForest_pca', RandomForest_grid_pca.best_estimator_, RandomForest_grid_pca.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with PCA:")
        print(RandomForest_grid_pca.best_params_)
        print("Best cross-validation accuracy with PCA:")
        print(RandomForest_grid_pca.best_score_)

    def RandomForest_sbs(self):
        # 랜덤 포레스트 모델 정의
        rf = RandomForestClassifier()

        # SBS 객체 생성
        sbs = SequentialFeatureSelector(rf, n_features_to_select=10, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', RandomForestClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
        }

        # 그리드 서치 객체 생성
        RandomForest_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy', n_jobs=-1)

        # 모델 학습
        RandomForest_grid_sbs.fit(self.X, self.y)
        
        # 최적의 모델 저장
        self.models.append(('RandomForest_sbs', RandomForest_grid_sbs.best_estimator_, RandomForest_grid_sbs.best_score_))
        self.save_model('RandomForest_sbs', ('RandomForest_sbs', RandomForest_grid_sbs.best_estimator_, RandomForest_grid_sbs.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters with SBS:")
        print(RandomForest_grid_sbs.best_params_)
        print("Best cross-validation accuracy with SBS:")
        print(RandomForest_grid_sbs.best_score_)

    def KNN(self):
        # 파이프라인 구성
        pipe = Pipeline([
            ('classifier', KNeighborsClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2],  # p=1: Manhattan, p=2: Euclidean
        }

        # 그리드 서치 객체 생성
        knn_grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

        # 모델 학습
        knn_grid.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('KNN', knn_grid.best_estimator_, knn_grid.best_score_))
        self.save_model('KNN',('KNN', knn_grid.best_estimator_, knn_grid.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters for KNN:")
        print(knn_grid.best_params_)
        print(knn_grid.best_score_)

    def KNN_pca(self):
        # 파이프라인 구성
        pipe_pca = Pipeline([
            ('pca', PCA()),
            ('classifier', KNeighborsClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_pca = {
            'pca__n_components': [x for x in range(1, self.n_features)],
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2],  # p=1: Manhattan, p=2: Euclidean
        }

        # 그리드 서치 객체 생성
        knn_grid_pca = GridSearchCV(pipe_pca, param_grid_pca, cv=5, scoring='accuracy')

        # 모델 학습
        knn_grid_pca.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('KNN_pca', knn_grid_pca.best_estimator_, knn_grid_pca.best_score_))
        self.save_model('KNN_pca',('KNN_pca', knn_grid_pca.best_estimator_, knn_grid_pca.best_score_))

        # 최적의 파라미터와 점수 출력
        print("Best parameters for KNN with PCA:")
        print(knn_grid_pca.best_params_)
        print(knn_grid_pca.best_score_)

    def KNN_sbs(self):
        # KNN 모델 정의
        knn = KNeighborsClassifier()

        # SBS 객체 생성 (Sequential Feature Selector)
        sbs = SequentialFeatureSelector(knn, n_features_to_select=8, direction='backward', cv=5)

        # 파이프라인 구성
        pipe_sbs = Pipeline([
            ('feature_selection', sbs),
            ('classifier', KNeighborsClassifier())
        ])

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid_sbs = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2],  # p=1: Manhattan, p=2: Euclidean
        }

        # 그리드 서치 객체 생성
        knn_grid_sbs = GridSearchCV(pipe_sbs, param_grid_sbs, cv=5, scoring='accuracy')

        # 모델 학습
        knn_grid_sbs.fit(self.X, self.y)

        # 최적의 모델 저장
        self.models.append(('KNN_sbs', knn_grid_sbs.best_estimator_, knn_grid_sbs.best_score_))
        self.save_model('KNN_sbs', ('KNN_sbs', knn_grid_sbs.best_estimator_, knn_grid_sbs.best_score_))
        # 최적의 파라미터와 점수 출력
        print("Best parameters for KNN with SBS:")
        print(knn_grid_sbs.best_params_)
        print(knn_grid_sbs.best_score_)




    def print_model_accuracies(self):
        print("Model Accuracies:")
        for name, model, acc in self.models:
            print(f"Model: {name}, Accuracy: {acc:.4f}")

    # 정확도를 기준으로 모델을 내림차순 정렬하는 메서드
    def sort_models_by_accuracy(self):
        self.models.sort(key=lambda x: x[2], reverse=True)

    # 상위 n개의 모델을 선택하여 반환하는 메서드
    def get_top_n_models(self, n):
        # 모델 정렬
        # 상위 n개의 모델 선택
        top_n_models = self.models[:n]
        for name, model, acc in top_n_models:
            print(f"Model: {name}, Accuracy: {acc:.4f}")
        return top_n_models
        
    def save_model(self, filename, model_tuple):
        model_name, model, model_score = model_tuple
        with open(filename, 'wb') as file:
            pickle.dump((model_name, model, model_score), file)
        print(f"모델 {model_name}이 {filename} 파일로 저장되었습니다.")

    # 파일에서 모델을 불러오는 메서드
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            model_tuple = pickle.load(file)
        model_name, model, model_score = model_tuple
        print(f"모델 {model_name}이 {filename} 파일에서 불러와졌습니다.")
        return model_tuple



# 앙상블 예측 메서드 추가
def ensemble_predict(models, X_train, y_train, X_test):

    # VotingClassifier를 위한 estimators 리스트 생성
    estimators = [(name, model) for name, model, best_score in models]

    # VotingClassifier 생성
    ensemble_model = VotingClassifier(estimators=estimators, voting='hard')

    # 앙상블 모델 학습 (각 개별 모델은 이미 학습되었으므로, 재학습 없이 fit 필요 없음)
    # 그러나 VotingClassifier의 fit 메서드를 호출해야 합니다.
    ensemble_model.fit(X_train, y_train)

    # 예측 수행
    predictions = ensemble_model.predict(X_test)

    return predictions