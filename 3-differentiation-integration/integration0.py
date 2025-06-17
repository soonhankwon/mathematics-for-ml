import sympy as sp

# x를 기호(Symbol)로 정의
x = sp.Symbol('x')

# 함수 f(x) = 3x^2 + 1 정의
f = 3 * x**2 + 1
# sympy의 integrate 함수를 사용하여 부정적분 계산
print(sp.integrate(3.0 * x**2 + 1, x))

# scipy의 수치적분 기능 불러오기
from scipy.integrate import quad

# 적분할 함수 정의
def f(x):
    return 3.0 * x**2 + 1

# quad 함수로 0에서 2까지 정적분 계산
# i[0]는 적분값, i[1]은 오차를 반환
i = quad(f, 0, 2)
print(i[0])