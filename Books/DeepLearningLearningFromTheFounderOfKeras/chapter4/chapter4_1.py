#todo 단순 홀드아웃 검증
#todo  - 데이터의 일정량을 테스트 세트로 떼어 놓고, 남은 데이터에서 훈련하고 테스트 세트로 평가한다.
#todo    정보 누설을 막기 위해 테스트 세트를 사용하여 모델을 튜닝해서는 안 되고, 검증 세트도 따로 떼어 놓아야 한다.
#todo  - 이 평가 방법은 단순해서 한 가지 단점이 있다.
#todo    데이터가 적을 때는 검증 세트와 테스트 세트의 샘플이 너무 적어 주어진 전체 데이터를 통계적으로 대표하지 못할 수 있다.
import numpy as np

num_validation_samples = 10000

data = []  # 특정 데이터 셋의 전체 데이터

np.random.shuffle(data)

# Train / Test 는 대략 7:3 비율로 분할
test_data = data[int(len(data) * 0.7):]
data = data[:int(len(data) * 0.7)]
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

training_data = data[:]

class get_model:
    def train(self, x):
        pass

    def evaluate(self, x):
        pass

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# 위에서 모델을 튜닝하고,
# 다시 훈련하고, 평가하고, 또 다시 튜닝하고...

model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)