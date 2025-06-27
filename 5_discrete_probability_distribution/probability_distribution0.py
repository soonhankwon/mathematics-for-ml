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

# 정규성 검정
x = stats.uniform().rvs(20) # 균일분포에서 표본 20개 추출
k2, p = stats.normaltest(x) # x에 대한 정규성 검정
print(p) # 정규성 검정 결과 해석
# p값이 0.05 이상이면 정규성을 만족한다고 볼 수 있다.
"""
0.011446870409049958
"""

# 카이제곱 검정
import scipy as sp
n = np.array([1,2,4,1,2,10]) # 주사위를 20번 던졌을때 1~6 사이의 눈이 나오는 빈도
print(sp.stats.chisquare(n)) # 귀무 가설: 각 눈의 빈도는 동일한 확률로 나옴
"""
Power_divergenceResult(statistic=17.799999999999997, pvalue=0.003207792034605283)
"""

# t검정
np.random.seed(0)
x1 = stats.norm(0, 1).rvs(10) # 평균이 0인 정규 분포에서 표본 10개 추출
x2 = stats.norm(1, 1).rvs(10) # 평균이 1인 정규 분포에서 표본 10개 추출
print(np.mean(x1), np.mean(x2)) # 두 랜덤 샘플의 평균 출력
# 0.7380231707288347 1.400646015162435

print(stats.ttest_ind(x1, x2)) # 두 집단의 모평균이 같다는 귀무 가설에 대해 t-검정
"""
TtestResult(statistic=-1.6868710732328158, pvalue=0.10888146383913824, df=18.0)
"""

# 쌍체 t검정
before = [68, 56, 70, 60, 65, 62, 63, 65, 64, 63]
after = [67, 55, 68, 58, 63, 60, 62, 64, 63, 62]
# 귀무 가설: 처치 전후로 통증의 차이가 없다.
# 대립 가설: 처치 전후의 통증 차이가 있다.
print(stats.ttest_rel(before, after))
"""
TtestResult(statistic=8.573214099741122, pvalue=1.2681848720135185e-05, df=9)
# 유의 수준이 5%라면 현재 p값이 유의 수준보다 작으므로 귀무 가설을 기각 -> 처치 전후의 통증 차이가 있다.
"""

# 쌍체 t검정
before = [2, 3, 2, 3, 2]
after = [9, 8, 9, 7 ,6]
# 귀무 가설: 광고 전후로 선호도 차이가 없다.
# 대립 가설: 광고 전후로 선호도 차이가 있다.
print(stats.ttest_rel(before, after))
"""
TtestResult(statistic=-7.961865632364446, pvalue=0.001348170975769803, df=4)
# 유의 수준이 5%라면 현재 p값이 유의 수준보다 작으므로 귀무 가설을 기각 -> 선호도 차이가 있다.
"""