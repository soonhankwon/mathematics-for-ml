# sympy 라이브러리
import sympy as sp

# 기호 변수 x 정의
x = sp.Symbol('x')

# 3x^2 + 1 함수를 x에 대해 미분
# 결과: 6x
print(sp.diff(3 * x**2 + 1, x))

# 미분된 함수에 x=2 대입
# 결과: 12.0
print(sp.diff(3 * x**2 + 1, x).subs(x, 2.0))

# 원본 함수 f(x) = 3x^2 + 1 정의
def f(x):
    return 3 * x**2 + 1

# 수치적 미분 함수
# 중앙 차분법을 이용한 미분 근사
# h: 미소 구간(기본값 1e-5)
def d(x, h=1e-5):
    return(f(x + h) - f(x - h)) / (2 * h)

# x=2에서의 수치적 미분값 출력
# 이론값 12와 거의 동일한 결과가 나옴
print(d(2.0))
