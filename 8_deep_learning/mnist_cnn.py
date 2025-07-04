# TensorFlow 라이브러리 임포트
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# CNN 구현에 필요한 keras 컴포넌트 임포트
from tensorflow.keras import layers, models

# MNIST 데이터셋 로드 및 훈련/테스트 데이터 분리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 이미지 데이터를 CNN 입력에 맞게 reshape (샘플 수, 높이, 너비, 채널)
# MNIST 이미지는 28x28 크기의 흑백 이미지 (채널=1)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀값을 0~1 사이로 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# Sequential 모델 생성
model = models.Sequential()

# CNN 레이어 구성
# 첫 번째 컨볼루션 레이어: 32개의 3x3 필터, ReLU 활성화 함수
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 2x2 맥스풀링으로 특성 맵의 크기를 절반으로 축소
model.add(layers.MaxPooling2D((2, 2)))
# 두 번째 컨볼루션 레이어: 64개의 3x3 필터
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# 세 번째 컨볼루션 레이어: 64개의 3x3 필터
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 특성 맵을 1차원으로 펼치는 Flatten 레이어
model.add(layers.Flatten())
# 완전연결층: 64개의 뉴런, ReLU 활성화 함수
model.add(layers.Dense(64, activation='relu'))
# 출력층: 10개의 뉴런 (0~9 숫자 분류), softmax 활성화 함수
model.add(layers.Dense(10, activation='softmax'))
# 모델 구조 출력
model.summary()

"""
 Total params: 93,322 (364.54 KB)
 Trainable params: 93,322 (364.54 KB)
 Non-trainable params: 0 (0.00 B)
"""

# 모델 컴파일
# - optimizer: Adam 옵티마이저 사용
# - loss: 다중 클래스 분류를 위한 sparse_categorical_crossentropy 손실 함수
# - metrics: 정확도(accuracy) 측정
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 - 5 에포크 동안 훈련
model.fit(train_images, train_labels, epochs=5)

"""
Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.8887 - loss: 0.3462     
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9845 - loss: 0.0505  
Epoch 3/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9897 - loss: 0.0351  
Epoch 4/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9922 - loss: 0.0240 
Epoch 5/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 6ms/step - accuracy: 0.9933 - loss: 0.0204 
313/313 - 1s - 2ms/step - accuracy: 0.9884 - loss: 0.0433 
"""

# 테스트 데이터로 모델 성능 평가 98.84%
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
"""
0.9883999824523926
"""
