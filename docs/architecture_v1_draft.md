# Face Realistic Swap v1 — Architecture Draft

상태: **Draft 0.2 — executable skeleton**  
최상위 기준: [`../PROJECT_GOAL.md`](../PROJECT_GOAL.md)

## 1. 한 줄 정의

원본 영상의 연기, 장면, 가림과 조명을 보존하면서 master의 얼굴 신원만
교체하는 **one-shot, feed-forward, video face-swap 시스템**을 만든다.

InSwapper는 최종 의존성이 아니라 동작·품질·속도의 고정 비교군이다.

## 2. 설계 결론

FRS-v1은 다음 세 원칙을 따른다.

1. 얼굴 전체 영상을 생성하지 않고 정렬된 얼굴 crop만 생성한다.
2. 원본 얼굴의 고주파 RGB를 generator의 내부 얼굴 영역으로 직접 skip하지
   않는다. 표정과 자세는 identity-neutral geometry로 전달한다.
3. generator는 교체 얼굴 RGB뿐 아니라 alpha, visibility, confidence를 함께
   예측하고, 원본 프레임과의 최종 결합은 별도 compositor가 담당한다.

```text
Master references (1~5)
    ├─ Global identity encoder ───────────────┐
    └─ Local reference encoder ──────────────┤
                                             │
Target frame                                 │
    ├─ tracker/alignment                     │
    ├─ geometry + expression encoder ────────┤
    ├─ illumination/context encoder ─────────┤
    └─ occlusion estimator ──────────────────┤
                                             ▼
                              Canonical Swap Core
                               ├─ swapped RGB
                               ├─ inner-face alpha
                               ├─ visibility
                               └─ confidence
                                             │
Previous temporal state ─────────────────────┤
                                             ▼
                              Original-preserving Compositor
                                             │
                                             ▼
                                      Final video frame
```

## 3. 연구 레퍼런스와 사용 경계

### 구조 레퍼런스

- [LivePortrait](https://liveportrait.github.io/): appearance feature volume,
  implicit motion, warping, SPADE decoder와 retargeting 구조의 구현 참고점이다.
  공식 보고 속도는 RTX 4090에서 프레임당 12.8ms다.
- [CanonSwap](https://arxiv.org/abs/2507.02691): motion을 제거한 canonical
  공간에서 identity를 바꾸고, 얼굴 일부에만 identity를 주입하는 Partial
  Identity Modulation 개념의 연구 참고점이다.
- [DynamicFace](https://arxiv.org/abs/2501.08553): pose, expression, geometry,
  identity를 분리한 3D facial conditions와 temporal layer의 연구 참고점이다.
- [LivingSwap](https://aim-uofa.github.io/LivingSwap/): 긴 영상에서 keyframe을
  identity anchor로 사용하고 원본 영상을 reference로 유지하는 장기 품질
  상한선이다.

### 경계

- CanonSwap 코드·가중치는 ResearchRAIL-M이므로 제품 코드에 복사하거나
  파생 의존성으로 넣지 않는다.
- DynamicFace와 LivingSwap은 공식 제품용 구현 기반으로 가정하지 않는다.
- LivePortrait 코드는 MIT지만 공식 배포본의 InsightFace 검출 가중치는
  비상업 조건이므로 사용 시 교체한다.
- [AuraFace v1](https://huggingface.co/fal/AuraFace-v1)은 Apache-2.0으로
  표시된 identity encoder 후보지만, 학습 데이터와 가중치 provenance에 대한
  공개 질의가 해소되기 전에는 `candidate` 상태로 둔다.
- 논문 아이디어의 독립 구현도 특허·데이터·모델 권리 검토를 대신하지 않는다.

## 4. 입력 계약

### 4.1 Master identity pack

```text
reference_rgb:       [B, N, 3, 256, 256], N=1..5
reference_valid:     [B, N]
reference_quality:   [B, N]
```

권장 사진은 정면 무표정 한 장만이 아니라 정면, 좌우 20~40도, 미소 또는
입 벌림을 포함한다. 각 reference는 동일한 5점 또는 dense landmark 기준으로
정렬하되 원본 종횡비와 crop transform을 기록한다.

### 4.2 Target frame pack

첫 구현 해상도는 256×256, 품질 검증 후 384×384로 올린다.

```text
target_crop:         [B, 3, H, W]
landmark_xyzw:       [B, 478, 4]
blendshape:          [B, 52]
head_pose:           [B, 6]       # rotation + translation/scale
geometry_maps:       [B, 16, H/4, W/4]
lowfreq_illumination:[B, 3, H/4, W/4]
context_ring:        [B, 3, H, W]
occlusion_mask:      [B, 1, H, W]
crop_transform:      [B, 2, 3]
timestamp/track_id
```

`geometry_maps`는 눈, 눈썹, 코, 바깥 입술, 안쪽 입술, 얼굴 윤곽과 normalized
depth를 그룹별 heatmap으로 rasterize한다. 478개 좌표를 MLP 하나에 넣는 현재
방식보다 공간적 대응을 직접 제공한다.

`lowfreq_illumination`은 target crop을 16~32px로 축소 후 복원한 값 또는
저주파 luminance/chroma다. 피부의 대략적인 밝기와 색은 전달하지만 원본의
눈·코·입 texture가 그대로 통과하지 않게 한다.

`context_ring`은 얼굴 내부가 제거된 머리카락, 귀, 목과 경계 주변 RGB다.
내부 얼굴로 이어지는 skip connection에는 반드시 binary/soft spatial gate를
적용한다.

## 5. 신경망 구성

### 5.1 Reference identity branch

두 종류의 신원 표현을 사용한다.

1. `GlobalIdentityEncoder`: 얼굴 인식용 512차원 normalized embedding
2. `LocalReferenceEncoder`: master의 세부 형태와 texture를 담는 4단계
   feature pyramid

여러 reference의 global embedding은 quality-weighted spherical mean으로
합치고, local feature는 target pose와 reference pose의 차이를 입력으로 하는
attention pooling으로 결합한다. 정면 한 장밖에 없으면 local feature의
신뢰도를 낮추고 global embedding 의존도를 높인다.

Global encoder는 교체 가능한 adapter interface로 만든다.

```python
class IdentityEncoderProtocol:
    def encode(self, aligned_faces, quality, valid_mask):
        # global_embedding, local_pyramid, confidence
        ...
```

AuraFace 사용 여부와 무관하게 generator의 입출력 계약은 바뀌지 않는다.

### 5.2 Target attribute branch

세 encoder가 서로 다른 정보를 처리한다.

- `GeometryEncoder`: geometry maps, blendshape, head pose
- `IlluminationEncoder`: low-frequency target color/lighting
- `ContextEncoder`: 내부 얼굴을 가린 context ring과 occlusion

Target 원본 RGB를 그대로 받는 U-Net skip은 만들지 않는다. 이것은 재구성은
쉽게 만들지만 target identity가 결과로 새는 가장 짧은 경로가 된다.

### 5.3 Canonical Swap Core

권장 초안은 2D/얕은-3D hybrid feed-forward generator다.

```text
GeometryEncoder
    -> canonical feature grid [B, 256, H/16, W/16]

Global identity
    -> modulation vectors for 6 residual blocks

Local identity pyramid
    -> gated cross-attention at H/16, H/8, H/4

Illumination/context
    -> spatial SPADE-like modulation

Decoder
    -> RGB residual / alpha / visibility / confidence
```

핵심 모듈은 `PartialIdentityModulation`이다.

```text
normalized feature
  + global identity affine
  + local identity cross-attention
  + target attribute spatial affine
  + learned spatial identity gate
```

identity gate는 눈·코·입·볼·턱 내부에서는 높고, 머리카락·목·배경과 target
occluder에서는 낮아야 한다. 단순 타원형 alpha와 달리 출력 alpha와 identity
gate는 서로 다른 head로 학습한다.

첫 모델 규모는 60~100M parameter를 상한으로 둔다. FP16, batch 1 기준 L40S
generator 목표 지연시간은 50ms 이하이며 전체 파이프라인은 100~160ms/frame을
목표로 한다.

### 5.4 Output heads

```text
face_rgb:       [B, 3, H, W]  # 새 신원의 내부 얼굴
alpha:          [B, 1, H, W]  # 원본과 합성할 영역
visibility:     [B, 1, H, W]  # 원본 occluder를 복원할 영역
confidence:     [B, 1]        # 실패/fallback 판단
temporal_state: feature map
```

Generator가 완성된 전체 프레임을 출력하도록 학습하지 않는다. 출력 RGB는
얼굴 crop 좌표계에 머물며 원본으로의 역변환은 compositor에서 수행한다.

## 6. 시간축 처리

시간축 기능은 정지 이미지 generator가 기본 합격선을 통과한 뒤 추가한다.

### Phase 1: model-free stabilization

- track 단위 crop transform Kalman/EMA smoothing
- landmark와 blendshape one-euro 또는 adaptive EMA smoothing
- scene cut, track loss, 큰 pose jump에서 상태 초기화
- alpha와 color transform의 optical-flow guided smoothing

### Phase 2: feature temporal module

최근 2~4프레임의 H/8 feature를 유지하는 작은 ConvGRU 또는 windowed temporal
attention을 decoder 앞에 둔다. RGB 프레임 전체를 재생성하는 video diffusion은
사용하지 않는다.

### Phase 3: keyframe anchor

긴 영상에서는 신뢰도가 높은 프레임을 keyframe으로 선택하고 그 프레임의
출력 identity embedding과 local appearance를 track anchor로 저장한다. 이후
프레임은 master reference뿐 아니라 anchor feature에도 조건화한다. 장면 전환
또는 재등장 시 track identity centroid로 anchor를 다시 연결한다.

## 7. Original-preserving compositor

Compositor는 모델 부속 기능이 아니라 제품 품질의 절반을 담당하는 독립 모듈로
취급한다.

처리 순서:

1. generator alpha를 얼굴 parsing/geometry prior와 결합한다.
2. target의 손, 안경, 머리카락 등 visibility mask를 alpha에서 제외한다.
3. 얼굴 내부에서 robust affine color transform 또는 저주파 gain/bias를 맞춘다.
4. crop을 원본 좌표로 inverse warp한다.
5. 경계는 distance-transform feather 또는 Laplacian pyramid로 합성한다.
6. 원본 프레임은 최종 alpha 밖에서 bit-exact하게 유지한다.
7. confidence가 낮으면 원본 유지가 아니라 정책에 따라 blur/mosaic 등의 안전
   fallback을 사용한다.

```text
visible_alpha = predicted_alpha * (1 - target_occlusion)
result = corrected_generated * visible_alpha
       + original_frame * (1 - visible_alpha)
```

눈과 입의 원본 픽셀 passthrough는 표정 보존용 ablation으로만 유지한다. 최종
익명화 모드에서는 원본 신원 누출 평가를 통과하지 못하면 사용하지 않는다.

## 8. 학습 데이터 계약

현재 1명·38장 데이터는 코드의 forward/backward 확인에만 사용한다. 범용
one-shot face swap을 학습할 수 있는 데이터가 아니다.

### 필요한 sample 단위

```text
identity_id
source_references[1..5]
target_frame or target_clip[5..16]
target_landmarks/blendshapes/pose
target_visibility/segmentation
same_identity flag
paired_ground_truth (optional but strongly preferred)
rights/license/provenance metadata
```

### 데이터 단계

1. **Self-reconstruction:** 같은 사람의 다른 프레임을 source/target으로 구성한다.
2. **Paired synthetic swap:** 동일 motion·camera·lighting을 서로 다른 합성
   identity로 렌더링해 정확한 교체 ground truth를 만든다.
3. **Unpaired real swap:** 동의받은 실제 다중 identity 데이터에 identity,
   motion, attribute consistency loss를 적용한다.
4. **Temporal clips:** 연속 5~16프레임과 occlusion·motion blur·side pose가
   포함된 clip으로 시간축 loss를 학습한다.

권장 현실성 게이트:

| 단계 | 최소 데이터 목적 | 의미 |
|---|---:|---|
| 구조 smoke test | 10 identities, 수천 프레임 | conditioning이 작동하는지 확인 |
| cross-ID PoC | 100+ identities, 5만~20만 프레임 | 신원 분리 가능성 확인 |
| one-shot 일반화 | 1,000+ identities, 100만+ diverse frames | 미등록 인물 적용 검증 |
| 제품 견고성 | 수백만 프레임 + hard cases | 드라마 조건 검증 |

이는 보장 수치가 아니라 개발 게이트다. LivePortrait가 약 6,900만 프레임의
mixed image-video 학습을 사용했다는 사실은 강한 일반화가 모델 구조만으로
생기지 않는다는 참고점이다.

## 9. 학습 단계와 손실

### Stage A — same-identity reconstruction

목표는 geometry와 appearance warping이 실제 얼굴을 복원하는지 확인하는 것이다.

```text
L_rec       Charbonnier/L1 inside visible face
L_gradient  edge/detail preservation
L_alpha     supervised/BCE + Dice
L_landmark  output landmark reprojection
L_pose      head-pose consistency
L_color     low-frequency illumination consistency
```

### Stage B — cross-identity paired training

합성 paired ground truth를 중심으로 identity injection을 학습한다.

```text
L_pair      swapped output vs paired ground truth
L_id        output vs master identity cosine/angular loss
L_motion    output vs target expression/gaze/lip geometry
L_target_id target identity adversarial suppression
L_region    non-face/context preservation
```

`L_target_id`는 target identity classifier의 정보를 generator가 이용하지 못하게
gradient reversal 또는 adversarial discriminator로 제한한다. 무작정 target RGB를
blur하는 것만으로 identity leakage가 해결된다고 가정하지 않는다.

### Stage C — unpaired realism

```text
L_adv       multi-scale face PatchGAN
L_feature   discriminator feature matching
L_cycle     swap-back consistency, 낮은 가중치
L_attr      illumination/occlusion/pose consistency
```

Cycle loss가 target identity를 보존하도록 유도할 수 있으므로 주 손실로 쓰지 않는다.

### Stage D — temporal fine-tuning

```text
L_warp      optical-flow warped adjacent output consistency
L_landmark_t adjacent motion derivative consistency
L_id_t      clip 내 identity embedding variance
L_alpha_t   warped alpha flicker
```

빠른 표정 변화와 scene cut에는 temporal loss mask를 꺼서 motion을 과도하게
부드럽게 만들지 않는다.

모든 pretrained loss network는 코드 라이선스, weight 라이선스와 학습 데이터
provenance를 통과해야 한다. 통과하지 못한 모델은 연구 평가에만 격리한다.

## 10. 추론 파이프라인

```text
1. master reference 등록 및 feature cache
2. 영상 decode + audio 분리
3. frame별 얼굴 detect/track
4. track별 alignment와 geometry 추출
5. FRS-v1 generator 추론
6. temporal state와 keyframe anchor 업데이트
7. occlusion-aware paste-back
8. 실패 frame fallback
9. video encode + 원본 audio remux
10. 품질/속도/실패 report 저장
```

한 track의 master feature는 한 번만 계산한다. 검출도 매 프레임 전체 해상도에서
반복하지 않고 tracker 예측과 주기적 재검출을 조합한다.

## 11. 정량 합격 기준

현재 L40S 기준선을 동일 입력과 동일 crop 조건으로 다시 측정한다.

| 항목 | MVP hard gate | 목표 |
|---|---:|---:|
| 처리 속도, 30fps 1인 영상 | 6 processing fps 이상 | 10 fps 이상 |
| 실시간 배수 | 5배 이하 | 3배 이하 |
| 검출된 정면/준측면 생성 성공률 | 99% 이상 | 99.5% 이상 |
| track 내 신원 일관성 | calibrated threshold 통과 | InSwapper 이상 |
| 원본 신원 누출 | calibrated threshold 미만 | InSwapper 이하 |
| 입·눈·pose 오차 | InSwapper보다 악화 금지 | InSwapper보다 개선 |
| temporal flicker | InSwapper보다 악화 금지 | 유의미하게 개선 |
| 사람 평가 | InSwapper와 동등 이상 | 다수 선호 |

얼굴 인식 cosine threshold는 임의의 상수로 고정하지 않고 실제 운영 영상으로
ROC/EER을 만들어 결정한다. 생성에 사용한 identity encoder와 평가 encoder는
분리한다.

## 12. 실패 정책

다음 조건은 confidence를 낮춘다.

- 얼굴 크기가 학습 범위보다 작음
- yaw/pitch가 학습 범위를 벗어남
- landmark/track이 순간 이동함
- 얼굴 가림 비율이 높음
- motion blur 또는 노출 손실이 큼
- output identity가 master와 불일치함
- original identity leakage가 높음

익명화가 제품 목적이면 실패 프레임에 원본 얼굴을 그대로 노출하면 안 된다.
운영 모드에서는 blur, mosaic 또는 승인된 정적 대체 마스크로 fallback한다.

## 13. 저장소 구현 계획

기존 `modeling/network.py`의 `target_lowpass`와 `motion_only` 모델은
`legacy_v0` 실험으로 동결한다. 새 코드는 다음 경계로 만든다.

```text
src/face_realistic/modeling/v1/
├── config.py
├── contracts.py
├── identity.py
├── geometry.py
├── attributes.py
├── modulation.py
├── generator.py
├── temporal.py
├── model.py
└── losses.py

src/face_realistic/compositing/
├── masks.py
├── color.py
├── temporal.py
└── pasteback.py

src/face_realistic/data/v1/
├── schema.py
├── frames.py
├── clips.py
└── validation.py

src/face_realistic/evaluation/v1/
├── identity.py
├── motion.py
├── temporal.py
├── quality.py
└── benchmark.py
```

## 14. 구현 순서

1. 데이터·tensor contract와 shape test를 먼저 작성한다.
2. compositor를 generator와 분리하고 InSwapper crop을 입력해 단독 검증한다.
3. geometry map rasterizer와 low-frequency/context 분리를 구현한다.
4. same-identity reconstruction generator를 구현한다.
5. 최소 10 identities로 source가 바뀌면 output identity가 바뀌는지 검증한다.
6. paired synthetic cross-ID 학습을 추가한다.
7. 100 identities 게이트를 통과한 뒤 temporal module을 추가한다.
8. 256px에서 정확성을 통과한 뒤 384px, FP16, ONNX/TensorRT를 진행한다.

첫 성공 조건은 예쁜 preview 한 장이 아니다. 동일 target에서 master reference만
바꿨을 때 표정·포즈·장면은 유지되고 결과 identity만 일관되게 바뀌어야 한다.

## 15. 보류한 결정

- Global identity encoder를 AuraFace로 확정할지 자체 학습할지
- 3D implicit keypoint를 직접 학습할지 MediaPipe geometry만 사용할지
- 256→384 super-resolution head를 generator 내부에 둘지 후처리로 둘지
- 얼굴 parsing ground truth를 어떤 권리 확보 데이터로 만들지
- 합성 paired identity를 생성할 3D asset/renderer의 라이선스와 규모

이 결정들은 데이터 provenance와 10-identity 실험 결과가 나오기 전에 확정하지
않는다.

## 16. 현재 구현 상태

2026-08-21 기준으로 다음 수직 골격을 구현했다.

- `modeling/v1/geometry.py`: 478 landmark를 16개 geometry map으로 rasterize
- `modeling/v1/conditioning.py`: low-frequency illumination과 얼굴 내부가 제거된
  context ring 생성
- `modeling/v1/identity.py`: 1~5개 reference의 품질·validity 가중 집계
- `modeling/v1/attributes.py`: geometry와 identity-suppressed appearance pyramid
- `modeling/v1/modulation.py`: canonical attention과 partial identity gate
- `modeling/v1/generator.py`: RGB, alpha, visibility, confidence head
- `modeling/v1/model.py`: 전체 입력·출력 계약과 forward 연결
- `compositing/`: occlusion-aware alpha, color match와 inverse-warp paste-back

기본 256px 구성은 22,874,385 parameter이며 무작위 초기화 상태에서 forward가
동작한다. 이 수치는 품질 검증이 아니라 모델 연결과 메모리 규모 확인 결과다.
현재 완료된 테스트는 shape, forward/backward, reference masking, target 내부 RGB
차단, occlusion 복원과 alpha 바깥 원본 픽셀 불변성이다.

아직 구현하지 않은 것은 multi-identity dataset, v1 training objective/runner,
학습 checkpoint, 실제 영상 inference adapter, learned temporal module과 정량 평가
runner다. 따라서 이 상태에서 영상을 생성해도 의미 있는 Face Swap은 나오지 않는다.
