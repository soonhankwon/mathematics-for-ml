import numpy as np              # 수치 계산용 라이브러리
import pandas as pd            # 데이터 처리용 라이브러리
from sklearn import linear_model        # 로지스틱 회귀 모델 포함
from sklearn import model_selection    # 데이터 분할용

# 신용 데이터셋 불러오기 (대출 상환 여부 예측용)
data = pd.read_csv('./creditset.csv')
print(data.shape)  # 데이터 크기 확인 (2000개 데이터, 6개 특성)
"""
(2000, 6)
"""

# 입력 변수(X)와 출력 변수(Y) 분리
# X: 소득(income), 나이(age), 대출금액(loan)
# Y: 10년내 채무불이행 여부 (0: 정상상환, 1: 채무불이행)
X = data.loc[:, ['income', 'age', 'loan']]
Y = data['default10yr']
print(X.shape)  # 입력 데이터 크기 확인
"""
(2000, 3)
"""

# 학습 데이터와 테스트 데이터 분할 (7:3 비율)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)  # 모델 학습

# 학습된 계수 출력 (각 특성이 채무불이행 확률에 미치는 영향)
print(model.coef_)  # [소득 계수, 나이 계수, 대출금액 계수]
"""
[[-2.43235035e-04 -3.49070376e-01  1.73338206e-03]]
# 소득과 나이가 증가할수록 채무불이행 확률 감소
# 대출금액이 증가할수록 채무불이행 확률 증가
"""

# 테스트 데이터에 대한 예측
Y_pred = model.predict(X_test)  # 예측 확률 계산
Y_pred2 = [0 if x < 0.5 else 1 for x in Y_pred]  # 0.5를 기준으로 이진 분류
Y_pred3 = Y_pred2 == Y_test  # 예측값과 실제값 비교
print(np.mean(Y_pred3 == Y_test))  # 정확도 출력
"""
0.155  # 약 15.5%의 정확도
"""

# 모델 성능 평가
from sklearn.metrics import classification_report, confusion_matrix

# 혼동 행렬 출력 (예측값과 실제값의 관계)
print(confusion_matrix(Y_test, Y_pred3))
# 분류 보고서 출력 (정밀도, 재현율, F1 점수 등)
print(classification_report(Y_test, Y_pred3))
"""
[[ 17 491]  # 진짜 음성(TN): 17, 거짓 양성(FP): 491
 [ 17  75]]  # 거짓 음성(FN): 17, 진짜 양성(TP): 75

              precision    recall  f1-score   support
           0       0.50      0.03      0.06       508  # 정상상환 클래스
           1       0.13      0.82      0.23        92  # 채무불이행 클래스

    accuracy                           0.15       600  # 전체 정확도
   macro avg       0.32      0.42      0.15       600  # 매크로 평균
weighted avg       0.44      0.15      0.09       600  # 가중 평균
"""
