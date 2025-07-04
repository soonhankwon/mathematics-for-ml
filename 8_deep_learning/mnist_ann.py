# TensorFlow 라이브러리 임포트
import tensorflow as tf

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
# 훈련용 데이터(x_train, y_train)와 테스트용 데이터(x_test, y_test) 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 픽셀값을 0~1 사이로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential API를 사용하여 신경망 모델 생성
model = tf.keras.Sequential(
    [
        # 입력층: 28x28 이미지를 1차원으로 펼침
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # 은닉층: 128개의 뉴런, ReLU 활성화 함수 사용
        tf.keras.layers.Dense(128, activation='relu'),
        # 과적합 방지를 위한 Dropout 층 (20% 비율로 뉴런을 무작위 비활성화)
        tf.keras.layers.Dropout(0,2),
        # 출력층: 10개의 뉴런 (0~9 숫자 분류), softmax 활성화 함수 사용
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

# 모델 컴파일
# - optimizer: Adam 옵티마이저 사용
# - loss: 다중 클래스 분류를 위한 sparse_categorical_crossentropy 손실 함수
# - metrics: 정확도(accuracy) 측정
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 - 5 에포크 동안 훈련
model.fit(x_train, y_train, epochs=5)
# 테스트 데이터로 모델 성능 평가
model.evaluate(x_test, y_test, verbose=2)
"""
 loss: 0.0449
313/313 - 0s - 417us/step - accuracy: 0.9771 - loss: 0.0797
"""