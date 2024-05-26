## 본 페이지는 Deep Fake audio를 검출하기 위한 기본 분석 및 ML 코드가 첨부 되어있습니다.
## 목차
1. [Dataset](##Dataset)
2. [Feature](##Feature)
3. [Result](##Result)


## Dataset
### Deepfake
https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition

- 데이터셋 구조
    
    ```python
    DEEP-VOICE 데이터셋은 실제 인간 음성 및 AI로 생성된 DeepFake 음성을 포함하고 있으며, Retrieval-based Voice Conversion을 통해 음성을 변환한 예제들을 포함하고 있습니다. 이 데이터셋은 음성이 AI에 의해 생성되었는지를 감지할 수 있는 머신러닝 모델 개발을 목적으로 합니다.
    
    배경 및 필요성
    최근 음성 도메인에서의 생성적 AI 기술은 음성 복제 및 실시간 음성 변환을 가능하게 합니다. 이러한 기술은 사생활 침해 및 오용의 가능성을 제기하며, 따라서 DeepFake 음성 변환을 실시간으로 감지할 필요성이 대두되고 있습니다.
    
    원본 오디오 파일 (AUDIO 디렉토리):
    
    REAL 및 FAKE 클래스 디렉토리 내에 오디오 파일이 정리되어 있습니다.
    파일명은 실제 음성을 제공한 화자와 변환된 음성을 나타냅니다. 예를 들어, "Obama-to-Biden"은 Barack Obama의 음성이 Joe Biden의 음성으로 변환되었음을 의미합니다.
    추출된 특징 데이터 (DATASET-balanced.csv 파일):
    
    이 데이터는 연구에서 사용된 것으로, 1초 단위 오디오 창에서 추출된 각 특징을 포함하고 있습니다.
    무작위 샘플링을 통해 균형을 맞춘 데이터셋입니다.
    ```
### ASVspoof2019
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

- 데이터셋 구조
    ```python
      ./ASVspoof2019_root
              --> LA  
                      --> ASVspoof2019_LA_asv_protocols
                      --> ASVspoof2019_LA_asv_scores
                --> ASVspoof2019_LA_cm_protocols
                      --> ASVspoof2019_LA_dev
                      --> ASVspoof2019_LA_eval
                --> ASVspoof2019_LA_train
                --> README.LA.txt
              --> PA 
                      --> ASVspoof2019_PA_asv_protocols
                      --> ASVspoof2019_PA_asv_scores
                --> ASVspoof2019_PA_cm_protocols
                      --> ASVspoof2019_PA_dev
                      --> ASVspoof2019_PA_eval
                --> ASVspoof2019_PA_train
                --> README.PA.txt
            --> asvspoof2019_evaluation_plan.pdf
            --> asvspoof2019_Interspeech2019_submission.pdf
            --> README.txt
    
    ```
---

## Feature
![Untitled (42)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/1550f363-16d9-48b2-bd48-14b2fbbeec0e)
### Mel-spac

<aside>
💡 **Mel Spectrogram은 오디오 신호를 Mel 스케일로 변환한 후, 스펙트로그램을 생성하는 방법 Mel 스케일은 인간의 청각이 선형적이지 않고 주파수에 따라 다르게 반응한다는 점을 반영한 것으로, 낮은 주파수에서는 민감하게, 높은 주파수에서는 덜 민감하게 반응**

</aside>

liborsa 사용

```python
fake_mfcc_librosa = librosa.feature.mfcc(y=fake_ad, sr=fake_sr)
```

직접 구현

```python
def melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    # STFT 계산
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    # 멜 필터 계산
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # 멜 스펙트로그램 계산
    mel_spect = np.dot(mel_filter, stft)
    # 로그 파워 스펙트로그램으로 변환
    log_mel_spect = np.log(mel_spect + 1e-6)
    return log_mel_spect
```
![Untitled (43)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/b106b400-ad67-4e9d-a9f6-bdf3a1c161a1)
---

### MFCC

<aside>
💡 **MFCC는 오디오 신호 처리에서 가장 널리 사용되는 특징점 추출 방법. 
MFCC는 신호를 Mel 스케일 필터 뱅크를 통과시킨 후, 로그 스펙트럼을 취하고, 이를 다시 DCT (Discrete Cosine Transform)를 통해 변환하여 얻는다.
MFCC는 음성 신호의 주파수 특성을 잘 포착하며, 음성 인식, 화자 식별 등 다양한 분야에서 활용.**

</aside>

liborsa 사용

```python
fake_mfcc_librosa = librosa.feature.mfcc(y=fake_ad, sr=fake_sr)
```

직접 구현

```python
# MFCC 계산 함수
def MFCC(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128):
    # STFT 계산
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    # 멜 필터 계산
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # 멜 스펙트로그램 계산
    mel_spect = np.dot(mel_filter, stft)
    # 로그 취하기
    log_mel_spect = np.log(mel_spect + 1e-6)
    # 로그 멜 스펙트로그램의 이산 코사인 변환 (DCT) 계산
    mfcc = scipy.fftpack.dct(log_mel_spect, type=2, axis=0, norm='ortho')[:n_mfcc]
    return mfcc
```
![Untitled (44)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/8c64635e-d4fe-4cf7-864e-c6d86d810282)
---

### LFCC

<aside>
💡 **LFCC는 선형 주파수 스케일을 사용하여 Cepstral Coefficients를 계산하는 방법. 
Cepstral 분석은 신호를 시간 영역에서 주파수 영역으로 변환한 후, 로그를 취하고, 다시 역변환하는 과정을 거친다. LFCC는 MFCC와 유사하지만, Mel 스케일 대신 선형 주파수 스케일을 사용한다는 차이가 존재. 이는 특정 분야에서 MFCC보다 유리할 수 있다.**

</aside>

liborsa 사용

```python
fake_lfcc_librosa = librosa.feature.lfcc(y=fake_ad, sr=fake_sr)
```

직접 구현

```python
# LFCC 계산 함수
def LFCC(y, sr, n_fft=2048, hop_length=512, n_filter=40, n_lfcc=13):
    # Short-Time Fourier Transform (STFT) 계산
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    # 선형 필터 계산
    filter_banks = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_filter, fmin=0, fmax=sr/2, norm=None)
    filter_banks = filter_banks / np.max(filter_banks, axis=-1)[:, None]  # 정규화
    # 파워 스펙트럼에 필터 뱅크 적용
    energy = np.dot(filter_banks, stft)
    # 로그 취하기
    log_energy = np.log(energy + 1e-6)
    # 로그 에너지의 이산 코사인 변환 (DCT) 계산
    lfcc = scipy.fftpack.dct(log_energy, type=2, axis=0, norm='ortho')[:n_lfcc]
    return lfcc
```
![Untitled (45)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/74ae6428-2697-4e15-9895-df3eb5da731c)
---

### Chroma

<aside>
💡 **Chroma Feature는 음악에서 화음이나 멜로디를 분석할 때 사용. 이는 12개의 다른 반음(semitone) 각각의 강도를 나타내는 벡터로, 한 옥타브(octave) 내의 모든 음표를 12개의 구간으로 나눈 것. Chroma Feature는 음악의 조화와 멜로디 패턴을 분석하는 데 유용하며, 음악 장르 분류나 커버 곡 식별 등에 활용.**

</aside>

liborsa 사용

```python
fake_chroma_librosa = librosa.feature.chroma_cqt(y=fake_ad, sr=fake_sr, bins_per_octave=36)
```

직접 구현

```python
# CHROMA 계산 함수
def CHROMA(y, sr, n_fft=2048, hop_length=512, n_chroma=12):
    # STFT 계산
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # 주파수 빈도에 해당하는 피치 클래스 빈도를 계산하는 크로마 필터 뱅크 생성
    chroma_filter = librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    # 크로마그램 계산
    chroma = np.dot(chroma_filter, stft)
    # 크로마 값을 정규화
    chroma = chroma / chroma.sum(axis=0, keepdims=True)
    return chroma
```
![Untitled (46)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/b94e0f44-6b9b-45f9-9b9b-36986415bc11)

## Result

### ASVspoof

- Simple ANN 모델 구성

```python
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        # 입력 데이터의 총 크기: 1*64*188 = 12032
        self.linear1 = nn.Linear(12032, 1024)  # 첫 번째 선형 레이어
        self.relu = nn.ReLU()  # 활성화 함수
        self.linear2 = nn.Linear(1024, 512)  # 두 번째 선형 레이어
        self.linear3 = nn.Linear(512, 256)  # 세 번째 선형 레이어
        self.linear4 = nn.Linear(256, 1)  # 출력 레이어
        self.sigmoid = nn.Sigmoid()  # 출력 활성화 함수

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)  # 이진 분류를 위한 시그모이드 함수
        return x
```

- Simplee CNN 모델 구성

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 첫 번째 Convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # 출력 크기: 32x64x188
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 출력 크기: 32x32x94 (Pooling)
        # 두 번째 Convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # 출력 크기: 64x32x94
        # 두 번째 Pooling을 거치면 64x16x47
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 47, 1024)  # 첫 번째 완전 연결 레이어
        self.fc2 = nn.Linear(1024, 512)  # 두 번째 완전 연결 레이어
        self.fc3 = nn.Linear(512, 256)  # 세 번째 완전 연결 레이어
        self.fc4 = nn.Linear(256, 1)  # 최종 출력 레이어
        self.sigmoid = nn.Sigmoid()  # 출력 활성화 함수(이진 분류)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 첫 번째 Convolutional + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # 두 번째 Convolutional + Pooling
        x = x.view(-1, 64 * 16 * 47)  # Flatten
        x = F.relu(self.fc1(x))  # 첫 번째 Fully connected layer
        x = F.relu(self.fc2(x))  # 두 번째 Fully connected layer
        x = F.relu(self.fc3(x))  # 세 번째 Fully connected layer
        x = self.fc4(x)  # 최종 출력 레이어
        x = self.sigmoid(x)  # 시그모이드 활성화 함수
        return x
```

- Resnet101 모델 구성

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet101 모델사전 훈련된 가중치 사용하며, 입력 채널 수를 1
        self.model = timm.create_model('resnet101', pretrained=True, in_chans=1)
        
        # 모델의 파라미터 중 일부를 고정
        for i, (name, param) in enumerate(list(self.model.named_parameters())[:39]):
            param.requires_grad = False

        # ResNet101 모델의 마지막 두 개의 레이어를 제외한 나머지 레이어들을 features로 정의
        self.features = nn.Sequential(*list(self.model.children())[:-2])

        # 새로운 레이어를 정의. AdaptiveAvgPool2d를 통해 출력의 공간 크기를 (1, 1)로 조정, Flatten 레이어를 사용하여 2D 텐서를 1D로 평탄화
        # fully connected 레이어와 시그모이드 활성화 함수로 이루어진 레이어를 정의
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 출력의 공간 크기를 (1, 1)로 조정
            nn.Flatten(),  # 2D 텐서를 1D로 평탄화
            nn.Linear(self.model.num_features, 1),  # fully connected 레이어를 정의
            nn.Sigmoid()  # 시그모이드 활성화 함수를 적용하여 이진 분류를 수행
        )

    def forward(self, inputs):
        # 입력 데이터를 ResNet101의 특성 추출기 부분을 통과
        x = self.features(inputs)
        # 특성 추출된 데이터를 새로운 레이어(custom_layers)를 통과시켜 최종 출력을 계산
        x = self.custom_layers(x)
        return x

```

| 모델 | 주요 특징 | 적합한 데이터 |
| --- | --- | --- |
| SimpleANN | - 완전 연결된 레이어로 구성됨 | - 단순한 구조의 입력 데이터 |
|  | - ReLU 활성화 함수 사용 | - 이미지나 시퀀스가 아닌 데이터 |
|  | - 이진 분류를 위해 시그모이드 함수 사용 |  |
| SimpleCNN | - 합성곱 레이어와 풀링 레이어 사용 | - 이미지 데이터 |
|  | - 완전 연결된 레이어로 분류 | - 공간적 특징 추출이 필요한 경우 |
|  | - 이미지 처리에 특히 효과적 |  |
| Model | - ResNet101 아키텍처 기반 | - 이미지 데이터 |
|  | - 전이 학습을 통해 성능 향상 가능 | - 대규모 이미지 데이터, 딥러닝 태스크에 적합 |
|  | - 이미지 특성 추출을 위해 사전 훈련된 모델 사용 |  |

| 모델 | 구조 | 파라미터 수 | 특징 |
| --- | --- | --- | --- |
| SimpleANN | Flatten -> Linear(12032, 1024) -> ReLU -> Linear(1024, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 1) -> Sigmoid | 약 12,350,721 | 간단한 완전 연결 구조, ReLU 활성화 함수 사용 |
| SimpleCNN | Conv2d(1, 32) -> ReLU -> MaxPool2d -> Conv2d(32, 64) -> ReLU -> MaxPool2d -> Linear(64 * 16 * 47, 1024) -> ReLU -> Linear(1024, 512) -> ReLU -> Linear(512, 256) -> Linear(256, 1) -> Sigmoid | 약 7,177,857 | Convolutional 레이어와 Max Pooling 사용, ReLU 활성화 함수 사용 |
| Model (ResNet101 기반) | ResNet101의 특성 추출기 부분 -> AdaptiveAvgPool2d -> Flatten -> Linear -> Sigmoid | 약 42,497,969 | 사전 훈련된 ResNet101 아키텍처 사용, 전이 학습에 적합 |

|  |  | Simple ANN | Simple CNN | Resnet101 튜닝 | Resnet202 |  |
| --- | --- | --- | --- | --- | --- | --- |
| Mel-spac | Test Accuarcy | 88 | 90 | 92 | 98 | 이건 좋음 |
|  | EER | 0.053 | 0.061 | 0.404 | 0.09 | EER이 정확하지않음 |
|  |  |  |  |  |  |  |
| MFCC | Test Accuarcy |  |  |  | 95 |  |
|  | EER |  |  |  | 0.08 |  |
|  |  |  |  |  |  |  |

![Untitled (48)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/da89a4d7-beaa-403a-825a-80224112604b)
![Untitled (49)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/4b14e42b-c27d-4f94-9595-e972e9cb2e0b)
![Untitled (50)](https://github.com/lexxsh/Deepfake-Audio/assets/110239629/ed451e2c-53a8-4570-aa6d-8f8040641faa)
