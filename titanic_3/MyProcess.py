import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import copy


class myProcessor:
    
    def __init__(self, _df):
        self.df = copy.copy(_df)

    # 정상 작동 확인.
    def _nan_process(self, remove=True):
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        # 결측치 처리 (옵션1, 결측치 행제거 / 옵션2, 결측치 평균값으로 메우기)
        if remove:
            self.df.dropna(subset=numeric_cols, inplace=True)
        else:
            # 평균값으로 결측치 메우기
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

    # 문제 해결.
    def _name_process(self):
        # Name : 호칭만 원핫 인코딩 (5개 인코딩)
        name_df = self.df['Name'].str.split(pat='[,.]', n=2, expand=True)
        self.df['honorific'] = name_df[1].str.strip()
        self.df['honorific'] = self.df['honorific'].apply(honorifics_classify)
        self.df.drop(columns=['Name'], inplace=True)

        # 원핫 인코딩
        df_honorific = self.df['honorific'].values
        honorific_ohe = OneHotEncoder()
        ohe_honorific = honorific_ohe.fit_transform(df_honorific.reshape(-1, 1)).toarray()
        honorific_list = honorific_ohe.categories_[0]
        df_ohe_honorific = pd.DataFrame(ohe_honorific, columns=honorific_list, index=self.df.index)

        # 축은 열 단위로 합쳐야 함 (axis=1)
        self.df = pd.concat([self.df, df_ohe_honorific], axis=1)
        self.df.drop(columns=['honorific'], inplace=True)

    # 문제없음
    def _remove(self):
        # Ticket, Cabin, PassengerId 삭제
        self.df.drop(columns=['Ticket', 'Cabin', 'PassengerId'], inplace=True)

    # 문제없다.
    #Parch 사용여부, Sibsp사용여부
    def _parch_sibsp_process(self, Parch_drop=True, SibSp_drop=True):
        # Parch 삭제 / 사용 선택
        if Parch_drop:
            self.df.drop(columns=['Parch'], inplace=True)

        # SibSp 삭제 / 사용 선택
        if SibSp_drop:
            self.df.drop(columns=['SibSp'], inplace=True)

    # 문제일 가능성 100퍼다.
    def _data_ohe_encoding(self, embarked=True): # 옵션으로 embark제거 또는 활용 선택.
        # Sex : 원핫 인코딩
        df_sex = self.df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.df['Sex'] = df_sex

        # Embarked : 원핫 인코딩
        if embarked:
            self.df.dropna(subset=['Embarked'], inplace=True)
            df_embarked = self.df['Embarked'].values
            embarked_ohe = OneHotEncoder()
            df_ohe_embarked = embarked_ohe.fit_transform(df_embarked.reshape(-1, 1)).toarray()
            embarked_list = embarked_ohe.categories_[0]
            df_ohe_embarked = pd.DataFrame(df_ohe_embarked, columns=embarked_list, index=self.df.index)
        self.df.drop(columns=['Embarked'], inplace=True)

        # Pclass : 원핫 인코딩
        df_pclass = self.df['Pclass'].values
        pclass_ohe = OneHotEncoder()
        df_ohe_pclass = pclass_ohe.fit_transform(df_pclass.reshape(-1, 1)).toarray()
        pclass_list = pclass_ohe.categories_[0]
        df_ohe_pclass = pd.DataFrame(df_ohe_pclass, columns=pclass_list, index=self.df.index)

        # Embarked가 True일 때만 df_ohe_embarked를 concat
        if embarked:
            self.df = pd.concat([self.df, df_ohe_embarked, df_ohe_pclass], axis=1)
        else:
            self.df = pd.concat([self.df, df_ohe_pclass], axis=1)


    def _age_process(self, used_age = False):
        if used_age:
            # 결측치의 행을 제거하고, Age를 10살 구간으로 나누고 그룹화
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=bins, labels=labels)
            self.df.drop(columns=['Age'], inplace=True)
        else:
            self.df.drop(columns=['Age'], inplace=True)

    def _fare_process(self):
        # Fare : 표준화 후 사용
        self.df['Fare'] = (self.df['Fare'] - self.df['Fare'].mean()) / self.df['Fare'].std()


    '''
    # 결측치 처리 (옵션1, 결측치 행제거 / 옵션2, 결측치 평균값으로 메우기)
    # Parch 사용여부, Sibsp사용여부
    # 옵션으로 embark제거 또는 활용 선택.
    '''
    def preprocess_df(self, _remove=True, _Parch=False, _SibSp=False, _Embarked=True):
        self.remove = _remove
        self.Parch = _Parch
        self.SibSp = _SibSp
        self.Embarked = _Embarked

        # 수치형 데이터에 대해서만 수행. 평균값으로 채우기 또는 행을 제거하기.
        self._nan_process(_remove)

        # Age, Ticket, Cabin, PassengerId 제거
        self._remove()          #no
        self._age_process() #ok

        # 수치화 및 인코딩, 표준화 완료
        self._data_ohe_encoding(_Embarked) #ok
        self._name_process()  #no
        self._fare_process()  #ok 
        self._parch_sibsp_process(_Parch, _SibSp) #ok

        # 수치형 데이터에 대해, 결측치 처리 수행 (행제거(train) 또는 평균값 채우기(test))
        
        return self.df
    
    def get_splited_dataset(self, _test_size = 0.3, _ramdom_state = 0):
        X, y = self.df.drop(columns=['Survived']), self.df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = _test_size,
                                                            random_state= _ramdom_state,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test
    
   


def honorifics_classify(honorific):
    honorific_list = ['Mr', 'Miss', 'Mrs', 'Master']
    return honorific if honorific in honorific_list else 'else'
