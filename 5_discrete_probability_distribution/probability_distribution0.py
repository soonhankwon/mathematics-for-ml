import numpy as np
from scipy import stats

# 난수 생성 시 재현성을 위해 시드값 설정
np.random.seed(0)

# 이항분포에서 10개의 난수 생성
# n=10 (시행횟수), p=0.5 (성공확률)인 이항분포
print(stats.binom(10, 0.5).rvs(10)) # [5 6 5 5 5 6 5 7 8 5]

# 표준정규분포(평균=0, 표준편차=1)에서 10개의 난수 생성
print(stats.norm().rvs(10))

# 균등분포(0과 1사이의 균일한 분포)에서 10개의 난수 생성
print(stats.uniform().rvs(10))

# 자유도가 2인 카이제곱분포에서 10개의 난수 생성
print(stats.chi2(df=2).rvs(10))