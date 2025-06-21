# sympy 라이브러리를 불러와 기호 미분을 수행
import sympy as sp
# x를 기호 변수로 정의
x = sp.Symbol('x')
# 3.0x^2 + 1 함수를 x에 대해 미분
print(sp.diff(3.0 * x**2 + 1, x))
# 6.0*x

# 수치해석을 위한 fsolve와 numpy를 불러옴
from scipy.optimize import fsolve
import numpy as np
# x + 3 = 0 형태의 1차 방정식을 정의
line = lambda x: x + 3
# 초기값 -2를 사용해 방정식의 해를 구함
soluition = fsolve(line, -2)
print(soluition)
# [-3.]

# 수치 적분을 위한 quad 함수를 불러옴
from scipy.integrate import quad
# cos(e^x)^2 함수를 정의
func = lambda x: np.cos(np.exp(x))**2
# 0부터 3까지 정의된 함수를 적분
solution = quad(func, 0, 3)
print(solution)
# (1.296467785724373, 1.3977971740145034e-09)
# quad 함수는 (적분값, 오차) 형태의 튜플을 반환



