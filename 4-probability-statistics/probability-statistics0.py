# 장바구니 데이터셋 정의
dataset = [
    ['Milk', 'Cookie', 'Apple', 'Beans', 'Eggs', 'Yogurt'],
    ['Coke', 'Cookie', 'Apple', 'Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Orange', 'Corn', 'Beans', 'Yogurt'],
    ['Corn', 'Cookie', 'Cookie', 'Beans', 'Ice cream', 'Eggs'],
]

print(type(dataset))
# list 타입 확인

# 데이터 전처리를 위해 pandas와 mlxtend 라이브러리 임포트
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

# 트랜잭션 데이터를 이진 행렬로 변환
te_ary = te.fit(dataset).transform(dataset)
"""
각 행은 하나의 거래를 나타내며, 각 열은 상품의 존재 여부를 True/False로 표시
"""

# 데이터프레임으로 변환
df = pd.DataFrame(te_ary, columns = te.columns_)
print(df)
"""
각 상품의 구매 여부를 True/False로 표시한 데이터프레임 출력
"""

# 연관 규칙 분석을 위한 라이브러리 임포트
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 최소 지지도 0.6으로 apriori 알고리즘 적용하여 빈발 아이템셋 찾기
print(apriori(df, min_support=0.6))
"""
숫자로 표현된 빈발 아이템셋과 각각의 지지도 출력
"""

# 상품명이 나오도록 출력
print(apriori(df, min_support=0.6, use_colnames=True))
"""
상품명으로 표현된 빈발 아이템셋과 각각의 지지도 출력
"""

# 제품 개수에 따른 필터링을 위해 length 추가
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
"""
각 아이템셋의 길이 정보가 추가된 빈발 아이템셋 출력
"""

# 패턴 내 아이템은 두 개이고, 최소 지지도는 60% 이상인 패턴만 필터링
print(frequent_itemsets[(frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.6)])
"""
두 개의 상품으로 구성된 빈발 아이템셋 출력
"""

# 신뢰도 기준으로 연관 규칙 생성 (최소 신뢰도 0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules)
"""
신뢰도가 0.7 이상인 연관 규칙들의 상세 정보 출력
(선행항, 결과항, 지지도, 신뢰도, 향상도 등)
"""

# 향상도 기준으로 연관 규칙 생성 (최소 향상도 1.2)
rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
print(rules2)
"""
향상도가 1.2 이상인 연관 규칙들의 상세 정보 출력
"""

# 복잡한 조건으로 규칙 필터링
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
print(rules[(rules['antecedent_len'] > 2) & (rules['confidence'] >= 0.75) & (rules['lift'] > 1.2)])
"""
선행항의 길이가 2 초과, 신뢰도 0.75 이상, 향상도 1.2 이상인 규칙 출력
"""