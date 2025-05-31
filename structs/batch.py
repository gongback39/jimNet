import numpy as np

class BatchLoader:
    def __init__(self, x, y, batch_size=2, shuffle=True):
        assert len(x) == len(y), "x와 y의 길이가 같아야 합니다." # 재데로된 데이터 셋이 들어왔는 지 확인
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size # 배치 크기
        self.shuffle = shuffle # 데이터 셔플 여부
        self.indices = np.arange(len(self.x)) # 인덱스 배열
        self.current = 0 # 현재 인덱스
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current = 0 # 현재 인덱스를 0으로 초기화
        # 데이터 셔플
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current >= len(self.x):
            raise StopIteration # 모든 배치를 사용한 경우 StopIteration 예외 발생

        # 배치 인덱스 계산
        start = self.current
        end = min(start + self.batch_size, len(self.x))
        batch_indices = self.indices[start:end]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        self.current = end # 현재 인덱스를 업데이트
        return batch_x, batch_y # 배치 반환 

    
    def __len__(self):
        # 전체 배치 개수 반환
        return int(np.ceil(len(self.x) / self.batch_size))
