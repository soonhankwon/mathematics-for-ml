# TensorFlow 라이브러리 임포트
import tensorflow as tf

# TensorFlow 기본 동작 테스트
hello = tf.constant('Hello, TensorFlow!')
print(hello)
"""
tf.Tensor(b'Hello, TensorFlow!', shape=(), dtype=string)
"""

import numpy as np

# 입력 데이터(x) 생성 - 2차원 좌표점 6개
x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
# 출력 데이터(y) 생성 - 각 좌표점에 대한 3개 클래스의 원-핫 인코딩
y_data = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1]])

# Sequential API를 사용하여 신경망 모델 생성
model = tf.keras.Sequential(
    [
        # 첫 번째 층: 10개의 뉴런, ReLU 활성화 함수 사용
        tf.keras.layers.Dense(10, activation='relu'),
        # 출력층: 3개의 뉴런 (3개 클래스 분류)
        tf.keras.layers.Dense(3)
    ]
)

# 모델 컴파일
# - optimizer: Adam 옵티마이저 사용
# - loss: 다중 클래스 분류를 위한 categorical_crossentropy 손실 함수
# - metrics: 정확도(accuracy) 측정
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 - 10 에포크 동안 훈련
model.fit(x_data, y_data, epochs=10)
"""
Epoch 1/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 219ms/step - accuracy: 0.8333 - loss: nan
"""
