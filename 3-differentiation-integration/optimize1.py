# 수치해석을 위한 scipy.optimize의 fsolve
from scipy.optimize import fsolve

# x + 3 = 0 형태의 1차 방정식을 정의
# lambda 함수를 사용한 간단한 선형 함수
line = lambda x: x + 3

# fsolve를 사용해 방정식의 해를 구함
# 초기값으로 -2를 사용
solution = fsolve(line, -2)

# 구한 해를 출력
print(solution)

# [-3.]