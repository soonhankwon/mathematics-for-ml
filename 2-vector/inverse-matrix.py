import numpy as np

# 2x2 행렬 생성
a = np.array([[1., 2.], [3.,4.]])

# 역행렬(inverse matrix) 계산
# 역행렬: 어떤 행렬 A와 곱했을 때 단위행렬이 되는 행렬
# A * A^(-1) = I (단위행렬)
inverse_matrix = np.linalg.inv(a)
print("역행렬:")
print(inverse_matrix)

# 행렬식(determinant) 계산
# 행렬식: 정사각행렬을 하나의 수로 대응시키는 값
# 행렬식이 0이 아닐 때만 역행렬이 존재
determinant = np.linalg.det(a)
print("\n행렬식:")
print(determinant)