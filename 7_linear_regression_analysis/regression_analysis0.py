import numpy as np

# 공부 시간(x)과 시험 점수(y) 데이터 생성
# 첫번째 열: 시험 점수(y), 두번째 열: 공부 시간(x)
data = np.array([
    [100, 30],
    [90, 26],
    [50, 15],
    [70, 20],
    [80, 22],
    [75, 23],
    [60, 15],
    [100, 35],
    [20, 2],
])

# 입력 데이터(x)와 출력 데이터(y) 분리
# reshape(-1,1)은 2차원 배열로 변환하기 위함
x = data[:, 1].reshape(-1, 1)
y = data[:, 0]

# matplotlib을 이용한 데이터 시각화
from matplotlib import pyplot as plt
plt.scatter(x, y)            # 산점도 그리기
plt.xlabel("study")          # x축 레이블 설정
plt.ylabel("score")          # y축 레이블 설정
plt.show()                   # 그래프 출력(선형관계)

# sklearn의 linear_model 모듈에서 LinearRegression 클래스 임포트
from sklearn import linear_model

# 선형 회귀 모델 객체 생성
regr = linear_model.LinearRegression()

# fit() 메서드로 데이터를 학습
# x: 입력 데이터(공부시간), y: 출력 데이터(시험점수)
regr.fit(x, y)

# 회귀 계수(기울기) 출력
print(regr.coef_)
"""
[2.62072585]
주당 공부시간이 한시간 늘 때마다 성적은 2.6점 상승 의미
"""

# sklearn.metrics에서 결정계수(R^2) 계산 함수 임포트
from sklearn.metrics import r2_score

# 학습된 모델로 예측값 계산
y_pred = regr.predict(x)

# 실제값과 예측값 사이의 결정계수 계산
# 1에 가까울수록 모델의 성능이 좋음을 의미
r2_score(y, y_pred)
plt.scatter(x, y, color="black")

# 예측된 회귀선은 파란색 실선으로 표시
plt.plot(x, y_pred, color="blue", linewidth=3)
plt.xlabel("study")
plt.ylabel("score")

# 그래프 출력
plt.show()