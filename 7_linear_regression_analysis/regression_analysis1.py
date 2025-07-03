import numpy as np              # 수치 계산용 라이브러리
import pandas as pd            # 데이터 처리용 라이브러리
from sklearn import linear_model        # 선형 회귀 모델
from sklearn import model_selection    # 데이터 분할
from sklearn import metrics           # 모델 평가 지표
from sklearn import preprocessing     # 데이터 전처리
from matplotlib import pyplot as plt  # 시각화

data = pd.read_csv("./Advertising.csv")
print(data.shape)  # 데이터 크기 출력 (행: 200, 열: 5)
"""
(200, 5)
"""

# TV 광고비와 매출액 간의 산점도 그리기
plt.scatter(data['TV'], data['Sales'])
plt.title('TV vs Sales')
plt.xlabel('TV')          # x축: TV 광고비
plt.ylabel('Sales')       # y축: 매출액
plt.show()

# 입력 변수(X)와 출력 변수(Y) 분리
X = data.loc[:, ['TV']]   # TV 광고비를 독립 변수로 선택
Y = data['Sales']         # 매출액을 종속 변수로 선택
print(X.shape)            # 독립 변수 데이터 크기 확인
"""
(200, 1)
"""

# 학습 데이터와 테스트 데이터 분할 (7:3 비율)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

# 선형 회귀 모델 생성 및 학습
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)  # 모델 학습

# 모델 성능 평가
print(regr.score(X_train, Y_train))  # R² 값 출력 (1에 가까울수록 좋음)
print(regr.coef_)  # 회귀 계수(기울기) 출력 - TV 광고비가 1단위 증가할 때 매출액 증가량
"""
0.5736021199591975  # R² 값이 0.57로 중간 정도의 설명력
[0.0464078]        # TV 광고비 1단위 증가시 매출액 0.046 증가
"""

# 테스트 데이터에 대한 예측값 계산
Y_pred = regr.predict(X_test)

# MSE(Mean Squared Error) 계산 - 예측값과 실제값의 차이를 제곱한 평균
# 값이 작을수록 예측이 정확함을 의미
print(np.mean((Y_pred - Y_test) ** 2))
"""
8.970991242413614  # MSE 값이 약 8.97로, 예측과 실제 매출액의 차이가 존재
"""

plt.scatter(X_test, Y_test, color='black')  # 실제 데이터 포인트 (검은색)
plt.plot(X_test, Y_pred, color="blue", linewidth=3)  # 예측된 회귀선 (파란색)
plt.xlabel('TV')      # x축: TV 광고비 
plt.ylabel('Sales')   # y축: 매출액
plt.show()           # 그래프 표시 - 회귀선과 실제 데이터의 관계 확인 가능