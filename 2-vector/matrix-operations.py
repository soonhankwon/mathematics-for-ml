# numpy: 수치 계산 라이브러리
import numpy as np
A = np.matrix([[3, 2, 4], [0, 4, 0], [0, 0, 5]])
B = np.matrix([[5, 0, 0], [3, 1, 0], [0, 2, 1]])

# 행렬의 차
print(A - B)
# 행렬의 곱
print(A * B)
# 행렬의 합
print(A + B)
