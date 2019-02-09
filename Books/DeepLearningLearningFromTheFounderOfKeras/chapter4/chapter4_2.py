#todo K-겹 교차 검증
#todo  - 데이터를 동일한 크기를 가진 K 개 분할로 나누고, 각 분할 i에 대해 남은 K - 1개의 분할로 모델을 훈련하고 분할 i에서 모델을 평가한다.
#todo    최종 점수는 이렇게 얻은 K개의 점수를 평균한다.
#todo    이 방법은 모델의 성능이 데이터 분할에 따라 편차가 클 때 도움이 되고, 홀드아웃 검증처럼 모델의 튜닝에 별개의 검증 세트를 사용한다.
import numpy as np

k = 4

data = []
test_data = data[int(len(data) * 0.7):]
data = data[:int(len(data) * 0.7)]
num_validation_samples = len(data) // k

class get_model:
    def train(self, x):
        pass

    def evaluation(self, x):
        pass

validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]

    model = get_model()  # 훈련되지 않은 새로운 모델 생성
    model.train(training_data)
    validation_score = model.evaluation(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)

model = get_model()
model.train(data)  # 테스트 데이터를 제외한 전체 데이터로 최종 모델을 훈련
test_score = model.evaluation(test_data)

#todo 셔플링을 사용한 반복 K-겹 교차 검증
#todo  - 이 방법은 비교적 가용 데이터가 적고 가능한 정확하게 모델을 평가하고자 할 때 사용한다.
#todo    캐글 경연에서는 이 방법이 아주 크게 도움이 된다.
#todo    이 방법은 K-겹 교차 검증을 여러 번 적용하되 K 개의 분할로 나누기 전에 매번 데이터를 무작위로 섞는다.
#todo    최종 점수는 모든 K-겹 교차 검증을 실행해서 얻은 점수의 평균이 된다.
#todo    결국 PxK(P는 반복 횟수)의 모델을 훈련하고 평가하므로 비용이 매우 많이 든다.