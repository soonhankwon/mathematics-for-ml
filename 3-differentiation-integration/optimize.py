from scipy.optimize import fsolve  # 방정식의 해를 찾기 위한 함수
import numpy as np

# x + 3 = 0 형태의 1차 방정식 정의
line = lambda x: x + 3

# fsolve를 사용하여 방정식의 해 구하기
# 첫 번째 인자: 방정식 함수
# 두 번째 인자: 초기 추측값 (-2)
solution = fsolve(line, -2)

# 결과 출력 (예상 결과: -3)
print(solution)