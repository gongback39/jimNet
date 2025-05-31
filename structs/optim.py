import numpy as np

# optimizer

#SGD
class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, weights, grad_weights, bias, grad_bias):
        """
        확률적 경사 하강법 (SGD) 수식:

            W ← W - η · ∇W
            b ← b - η · ∇b

        - η (lr): 학습률
        - ∇W, ∇b: 각각 가중치 및 편향에 대한 손실 함수의 기울기
        """
        weights -= self.lr * grad_weights
        bias -= self.lr * grad_bias
        return weights, bias

#Adam
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.states = {}  # 계층별 상태 저장 딕셔너리

    def _init_state(self, weights, bias):
        # 하나의 인스턴스로 여러 계층이 공유할 수  하기 때문에 각 파라미터의 상태를 저장할 필요가 있음.
        # 각 파라미터마다 고유 ID 생성 (메모리 주소 기반)
        w_id = id(weights)
        b_id = id(bias)
        
        if w_id not in self.states:
            self.states[w_id] = {
                't': 0,
                'm_w': np.zeros_like(weights),
                'v_w': np.zeros_like(weights)
            }
        if b_id not in self.states:
            self.states[b_id] = {
                't': 0,
                'm_b': np.zeros_like(bias),
                'v_b': np.zeros_like(bias)
            }

    def step(self, weights, grad_weights, bias, grad_bias):
        """
        Adam 옵티마이저 수식:

        1. 1차 모멘텀(평균):
            mₜ ← β₁ · mₜ₋₁ + (1 - β₁) · ∇θₜ

        2. 2차 모멘텀(분산):
            vₜ ← β₂ · vₜ₋₁ + (1 - β₂) · (∇θₜ)²

        3. 바이어스 보정:
            m̂ₜ = mₜ / (1 - β₁ᵗ)
            v̂ₜ = vₜ / (1 - β₂ᵗ)

        4. 파라미터 업데이트:
            θₜ ← θₜ₋₁ - η · m̂ₜ / (√(v̂ₜ) + ε)

        - θ: 파라미터 (W 또는 b)
        - ∇θₜ: 해당 파라미터에 대한 기울기
        - β₁, β₂: 각각 1차, 2차 모멘텀 계수
        - η (lr): 학습률
        - ε: 안정성 확보용 작은 수 (division by zero 방지)
        """
        self._init_state(weights, bias)
        w_id = id(weights)
        b_id = id(bias)
        
        # 가중치 업데이트
        self.states[w_id]['t'] += 1
        self.states[w_id]['m_w'] = self.beta1 * self.states[w_id]['m_w'] + (1 - self.beta1) * grad_weights
        self.states[w_id]['v_w'] = self.beta2 * self.states[w_id]['v_w'] + (1 - self.beta2) * (grad_weights**2)
        m_w_hat = self.states[w_id]['m_w'] / (1 - self.beta1**self.states[w_id]['t'])
        v_w_hat = self.states[w_id]['v_w'] / (1 - self.beta2**self.states[w_id]['t'])
        weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        
        # 편향 업데이트
        self.states[b_id]['t'] += 1
        self.states[b_id]['m_b'] = self.beta1 * self.states[b_id]['m_b'] + (1 - self.beta1) * grad_bias
        self.states[b_id]['v_b'] = self.beta2 * self.states[b_id]['v_b'] + (1 - self.beta2) * (grad_bias**2)
        m_b_hat = self.states[b_id]['m_b'] / (1 - self.beta1**self.states[b_id]['t'])
        v_b_hat = self.states[b_id]['v_b'] / (1 - self.beta2**self.states[b_id]['t'])
        bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
        
        return weights, bias

"""
class Adam():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0
        self.t = 0

    def step(self, weights, grad_weights, bias, grad_bias):
        self.t += 1

        # Weights
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_weights
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_weights ** 2)
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

        # Bias
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_bias
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_bias ** 2)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        return weights, bias
"""