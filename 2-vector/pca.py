# 주성분 분석(PCA) - 고차원 데이터를 저차원으로 축소하는 기법
from sklearn.decomposition import PCA
import numpy as np

# 10개의 3차원 데이터 포인트로 구성된 행렬
M = np.array(
    [[-1, -1, -1], [-2, -1, 2], [-3, -2, 0], [1, 1, 2], [2, 1, 1], 
    [3, 2, 4], [2, 0, 3], [3, 5, 1], [4, 2, 3], [3, 3, 2]]
)
print("원본 데이터 형태:", M.shape)  # (10, 3) 출력

# 주성분을 2개로 지정하여 3차원 -> 2차원으로 축소
pca = PCA(n_components=2)

# 데이터에 PCA 적용
pca.fit(M)

# PCA 모델 파라미터 확인
print("\nPCA 모델 설정:")
print(PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
svd_solver='auto', tol=0.0, whiten=False))

# 데이터를 주성분 공간으로 변환
PC = pca.transform(M)
print("\n차원 축소된 데이터 형태:", PC.shape)  # (10, 2) 출력

# 분산 설명 비율 확인 - 각 주성분이 원본 데이터의 분산을 얼마나 설명하는지
print("\n각 주성분의 분산 설명 비율:", pca.explained_variance_ratio_)

# 공분산 행렬의 고윳값(w)과 고유벡터(V) 계산
# 고윳값은 해당 방향의 분산을, 고유벡터는 주성분의 방향을 나타냄
w, V = np.linalg.eig(pca.get_covariance())
print("\n고윳값:", w)
print("고유벡터:\n", V)

# 원래 데이터를 주성분으로 표현
# V.T는 기저변환 행렬
transformed_data = V.T.dot(M.T).T
print("\n주성분으로 표현된 데이터:\n", transformed_data)