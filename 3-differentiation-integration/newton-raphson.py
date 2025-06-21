def f(x): # 함수 정의
    return (x**3 - 1) # x = 1에서 하나의 해 존재

from scipy.optimize import newton

# x축의 1.5값 지점에서 시작해서 뉴턴랩슨 메서드로 x의 해를 구함
root = newton(f, 1.5)
print(root)

# 1.0000000000000016

root = newton(f, 1.5, fprime= lambda x: 3 * x**2)
print(root)

# 1.0