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