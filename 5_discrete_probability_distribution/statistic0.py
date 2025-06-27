import numpy as np

# 0부터 9까지의 실수 배열
x = np.arange(10.0)
print(x.mean())  # x의 평균 4.5

# 0부터 19까지의 실수 배열을 4x5 행렬로 재구성
numbers = np.arange(20.0)
x = np.reshape(numbers, (4,5))
print(np.mean(x, 0))  # 각 열의 평균값 계산 [4.5, 5.5, 6.5, 7.5, 8.5]

# 표준편차 계산
print(np.std(x)) # 2.8722813232690143

# 분산 계산 
print(np.var(x)) # 8.25

# 3x4 크기의 정규분포 난수 행렬 생성
x = np.random.randn(3, 4)
print(np.corrcoef(x))  # x의 행별로 피어슨 상관관계 계산
print(np.corrcoef(x[0], x[1]))  # 첫 번째와 두 번째 행의 상관관계 계산

# 열을 변수로 간주하여 공분산 행렬 계산 (기본값)
print(np.cov(x, rowvar=False))

# 행을 변수로 간주하여 공분산 행렬 계산
print(np.cov(x, rowvar=True))