# Face Realistic

MediaPipe 기반 얼굴 퍼포먼스 추적과 실사 얼굴 가명화를 실험하기 위한 MVP입니다. 현재 1단계는 영상에서 478개 얼굴 랜드마크, 52개 블렌드셰이프, 4x4 얼굴 변환행렬과 머리 자세를 프레임별로 추출합니다.

## 빠른 시작

Python 패키지와 개발 의존성을 프로젝트 전용 `.venv`에 설치합니다.

```bash
uv sync --extra dev
```

기본 명령은 `assets/source/swap2.mp4`의 앞 3초를 처리합니다. 모델이 없으면 공식 MediaPipe Face Landmarker 모델을 `models/`에 자동으로 내려받습니다.

```bash
uv run python main.py
```

생성 결과:

- `outputs/tracking.jsonl`: 프레임별 타임스탬프, 추론 시간, 랜드마크, 블렌드셰이프, 변환행렬, 머리 자세
- `outputs/tracking.summary.json`: 검출률, 처리 FPS, 실시간 배수 요약
- `outputs/tracking_preview.mp4`: 랜드마크와 주요 블렌드셰이프 확인 영상(현재 오디오는 포함하지 않음)

전체 영상을 처리하려면 `--max-seconds 0`을 사용합니다.

```bash
uv run python main.py --max-seconds 0
```

다른 입력이나 출력 경로도 지정할 수 있습니다.

```bash
uv run python main.py \
  --input assets/source/example.mp4 \
  --output-jsonl outputs/example.jsonl \
  --output-preview outputs/example_preview.mp4
```

미리보기 없이 추적 데이터만 생성하려면 다음과 같이 실행합니다.

```bash
uv run python main.py --no-preview
```

테스트:

```bash
uv run pytest -q
```

## 대체 ID 등록

`assets/identities/person_01/`의 얼굴 사진을 검출하고 512×512로 정렬한 뒤 품질 manifest를 생성합니다.

```bash
uv run identity-register
```

생성 결과:

- `outputs/identity_registry/person_01/aligned/`: 정렬된 얼굴 이미지
- `outputs/identity_registry/person_01/manifest.json`: 파일 해시, 검출 결과, 머리 자세, 블렌드셰이프, 품질 점수와 권장 참조 이미지
- `outputs/identity_registry/person_01/contact_sheet.jpg`: 전체 정렬 결과와 품질 점수 확인표

품질 점수는 선명도, 노출, 정면성, 얼굴 면적을 합친 등록용 휴리스틱입니다. 얼굴 교체 결과의 최종 품질 점수나 익명성 보장을 의미하지 않습니다.

## Identity 임베딩

OpenCV YuNet으로 얼굴을 검출·정렬하고 SFace의 128차원 임베딩을 생성합니다. 권장 참조 이미지 8장의 품질 가중 평균을 `person_01`의 identity centroid로 사용하며, 기본 설정은 원본 영상 앞 3초의 12개 프레임과도 비교합니다.

```bash
uv run identity-embed
```

생성 결과:

- `outputs/identity_registry/person_01/identity_centroid.npy`: 대체 ID 대표 임베딩
- `outputs/identity_registry/person_01/source_centroid.npy`: 원본 영상 샘플 대표 임베딩
- `outputs/identity_registry/person_01/identity_profile.json`: 이미지별 임베딩·centroid 유사도와 원본 비교 결과

현재 샘플의 기준 결과는 대체 ID 38/38장과 원본 12/12프레임 임베딩 성공, 원본과 대체 ID의 centroid cosine similarity `0.235`입니다. SFace가 LFW에서 보고한 참고 임계값 `0.363`보다 낮아 진단상 서로 다른 identity입니다.

이 임계값은 현재 카메라·압축·인구집단에 맞게 보정된 운영 기준이 아닙니다. 익명성 평가 단계에서는 별도 얼굴 인식 모델을 추가하고 자체 데이터의 ROC를 이용해 임계값을 정해야 합니다. SFace 및 공개 벤치마크 정보는 [OpenCV 얼굴 인식 문서](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)를 참고하세요.

## Face swap 기준선

앞 3초의 가장 큰 얼굴을 `person_01/front_neutral.png`의 identity로 교체하고 원본 오디오를 다시 합칩니다.

필요 파일:

```text
assets/source/swap2.mp4
assets/identities/person_01/front_neutral.png
models/inswapper_128.onnx
```

InSwapper 가중치는 저장소에서 자동 다운로드하지 않습니다. InsightFace에서 별도로 허가받거나 연구 목적으로 적법하게 확보한 모델을 `models/inswapper_128.onnx`에 배치해야 합니다. InsightFace 사전학습 모델은 별도 라이선스가 없다면 비상업 연구용입니다.

```bash
uv sync --extra dev --extra swap --extra swap-cpu
uv run face-swap --max-seconds 3 --provider auto
```

출력:

- `outputs/swap_baseline.mp4`: 오디오가 포함된 3초 기준 영상
- `outputs/swap_baseline.json`: 프레임별 검출·처리시간과 전체 성능 요약

### EC2 설치 및 실행

Ubuntu 또는 Amazon Linux EC2에 프로젝트와 로컬에서 제외된 미디어·모델 파일을 업로드한 뒤 실행합니다.

```bash
FACE_SWAP_DEVICE=cpu bash scripts/setup_ec2.sh
bash scripts/run_swap_ec2.sh
```

NVIDIA GPU 인스턴스에서는 `nvidia-smi`로 드라이버가 정상인지 확인한 뒤 GPU extra로 설치합니다.

```bash
nvidia-smi
FACE_SWAP_DEVICE=gpu bash scripts/setup_ec2.sh
FACE_SWAP_PROVIDER=cuda bash scripts/run_swap_ec2.sh
```

CUDA 및 ONNX Runtime 버전 호환성은 EC2의 NVIDIA 드라이버 이미지에 맞춰야 합니다. CPU 인스턴스에서는 기본 `--provider auto`가 CPU를 선택합니다.

### 원본 표정 픽셀 보존 실험

InSwapper가 약화한 입 모양과 눈 깜빡임을 복구하기 위해 MediaPipe 랜드마크로 원본 눈·눈꺼풀·입술·입 내부를 soft mask 합성하는 실험입니다. 기존 swap 결과를 덮지 않습니다.

```bash
FACE_SWAP_PROVIDER=cuda bash scripts/run_passthrough_ec2.sh
```

출력은 `outputs/swap_passthrough.mp4`와 `outputs/swap_passthrough.json`입니다. 기본 보존 범위는 눈 `1.15`, 입 `1.35`, feather는 검출 얼굴 너비의 `1.2%`입니다. 면적을 키우면 표정과 가림은 더 보존되지만 원본 신원 누출 위험도 커집니다.

```bash
PASSTHROUGH_EYE_EXPANSION=1.1 \
PASSTHROUGH_MOUTH_EXPANSION=1.2 \
PASSTHROUGH_FEATHER_RATIO=0.01 \
FACE_SWAP_PROVIDER=cuda bash scripts/run_passthrough_ec2.sh
```

이 방식은 최종 익명화 해법이 아니라 원본 픽셀 보존 가설을 검증하는 통제 실험입니다. 결과는 생성 모델과 다른 얼굴 인식기로 원본 신원 유사도를 반드시 다시 측정해야 합니다.

## LivePortrait 표정 보존 기준선

InSwapper 결과에서 줄어든 입 벌림과 표정 강도를 검증하기 위해, 공식 LivePortrait로 대체 얼굴 이미지를 원본 영상의 움직임으로 구동합니다. 기존 프로젝트 `.venv`와 분리된 `third_party/LivePortrait/.venv`를 사용하므로 같은 EC2의 다른 프로젝트 패키지에는 영향을 주지 않습니다.

EC2에서 코드를 `git pull`한 다음 최초 한 번만 설치합니다. PyTorch CUDA 패키지와 모델 가중치를 내려받기 때문에 시간이 걸릴 수 있습니다.

```bash
cd ~/face_realistic
bash scripts/setup_liveportrait_ec2.sh
```

설치 스크립트는 Python 3.10 전용 환경을 만들고 공식 LivePortrait 저장소와 가중치를 준비한 뒤 CUDA 인식 여부를 검사합니다. 시스템 NVIDIA 드라이버는 변경하지 않습니다.

앞 3초 기준선을 실행합니다.

```bash
bash scripts/run_liveportrait_ec2.sh
```

생성 결과:

- `outputs/liveportrait_exp_baseline.mp4`: 머리 pose/scale은 고정하고 원본의 표정만 전달한 대체 얼굴 영상과 원본 오디오
- `outputs/liveportrait_exp_baseline.json`: 처리시간, 실시간 배수, FPS·해상도·오디오 유무
- `outputs/liveportrait_work/`: 잘라낸 driving clip, motion template, 공식 원본 출력

표정 강도가 여전히 약할 때만 `driving_multiplier`를 `1.1`처럼 조금 높여 두 번째 비교군을 만듭니다.

```bash
LIVEPORTRAIT_DRIVING_MULTIPLIER=1.1 \
LIVEPORTRAIT_OUTPUT=outputs/liveportrait_m110.mp4 \
LIVEPORTRAIT_REPORT=outputs/liveportrait_m110.json \
bash scripts/run_liveportrait_ec2.sh
```

기본값은 relative motion, `expression-friendly`, 표정 영역(`exp`), driving-video crop입니다. `exp` 실험에서는 대체 얼굴의 머리 pose/scale을 고정해 이전 `all` 실험에서 보인 얼굴 수축·윤곽 왜곡의 원인을 분리합니다. 이 결과는 아직 원본 프레임에 얼굴만 합성한 최종 face swap이 아니라 표정 전달 성능을 확인하는 중간 기준선입니다. 다음 단계에서 원본 MediaPipe pose를 이용한 paste-back, 가림 복원, 경계·조명 보정을 결합합니다.

이전 `all` 조건을 재현하려면 다음 환경변수를 사용합니다.

```bash
LIVEPORTRAIT_ANIMATION_REGION=all \
LIVEPORTRAIT_OUTPUT=outputs/liveportrait_all_baseline.mp4 \
LIVEPORTRAIT_REPORT=outputs/liveportrait_all_baseline.json \
bash scripts/run_liveportrait_ec2.sh
```

LivePortrait 코드는 MIT 라이선스지만 기본 얼굴 검출에 포함된 InsightFace 가중치는 비상업 연구 조건입니다. 상용화 시에는 허가된 검출 모델로 교체해야 합니다.

## JSONL 레코드

한 줄이 하나의 영상 프레임입니다. 좌표는 MediaPipe의 정규화 좌표이며, 얼굴이 검출되지 않은 프레임은 `detected: false`, `faces: []`로 기록합니다.

```json
{
  "schema_version": 1,
  "frame_index": 0,
  "timestamp_ms": 0,
  "detected": true,
  "inference_ms": 8.3,
  "faces": [
    {
      "landmarks": [[0.5, 0.4, -0.03]],
      "blendshapes": {"jawOpen": 0.42},
      "transformation_matrix": [[1.0, 0.0, 0.0, 0.0]],
      "head_pose": {
        "euler_xyz_deg": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0]
      }
    }
  ]
}
```

## 프로젝트 구조

```text
.
├── main.py
├── assets/                    # 샘플 이미지 등 정적 리소스
├── configs/                   # 실행 및 모델 설정
├── models/                    # 로컬 모델 가중치(버전 관리 제외)
├── outputs/                   # 추적·렌더링 결과(버전 관리 제외)
├── src/face_realistic/
│   ├── tracking/              # 검출, 랜드마크, 트래킹
│   ├── identity/              # 임베딩, 클러스터링, ID 배정
│   ├── performance/           # 표정, 시선, 포즈
│   ├── swap/                  # 얼굴 교체
│   ├── relighting/            # 조명·색·경계 보정
│   ├── evaluation/            # 품질·익명성·속도 평가
│   └── io/                    # 영상 및 결과 입출력
└── tests/
```

## 현재 범위

현재 구현은 단일 얼굴의 퍼포먼스 신호 추출 기준선입니다. 얼굴 임베딩, 인물 클러스터링, 대체 ID 생성·선택, face swap, 리라이팅은 다음 단계에서 이 추적 결과 위에 추가합니다.

입력 영상과 대체 얼굴은 사용 권리와 당사자 동의를 확보한 자료만 사용하세요. 모델 파일, 원본 영상, 생성 결과는 기본적으로 Git에서 제외됩니다.
