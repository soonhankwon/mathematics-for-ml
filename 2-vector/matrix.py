from scipy import linalg, sparse
import numpy as np

# 2x2 크기의 랜덤한 행렬 A 생성 (np.matrix 사용)
A = np.matrix(np.random.random((2, 2)))

# 2x2 크기의 랜덤한 배열 b 생성
b = np.random.random((2, 2))

# 배열 b를 행렬 B로 변환 (np.asmatrix 사용)
B = np.asmatrix(b)

# 10x5 크기의 랜덤한 행렬 C 생성
C = np.asmatrix(np.random.random((10 , 5)))

# 2x2 크기의 고정된 값을 가진 행렬 D 생성
D = np.asmatrix([[3, 4], [5, 6]])

# 생성된 모든 행렬 출력
print(A)
print(B) 
print(C)
print(D)

# A 행렬의 역행렬 출력
print(A.I)
# A 행렬의 행렬식(determinant) 계산 및 출력
print(linalg.det(A))

# 행렬 A와 D의 덧셈 연산 결과 출력
print(np.add(A, D))

# 행렬 A와 D의 뺄셈 연산 결과 출력 
print(np.subtract(A, D))

# 행렬 A와 D의 요소별 곱셈(Hadamard product) 결과 출력
print(np.multiply(A, D))

# 행렬 A와 D의 요소별 나눗셈 결과 출력
print(np.divide(A, D))

# 행렬 D와 B의 행렬 곱셈 결과 출력 (@ 연산자 사용)
print(D@B)
# 행렬 D와 B의 행렬 곱셈 결과 출력 (np.dot 함수 사용)
print(np.dot(D,B))

# 2x2 단위행렬(identity matrix) 생성 및 출력
G = np.asmatrix(np.identity(2))
print(G)
