# Clean-room face-swap architecture

> 이 문서는 실패 원인 추적을 위한 v0 기록이다. 현재 설계 초안은
> [`architecture_v1_draft.md`](architecture_v1_draft.md)를 따른다.

## Product goal

모든 아키텍처 결정은 [`../PROJECT_GOAL.md`](../PROJECT_GOAL.md)에 정의된
제품 목표와 합격 조건을 따른다. 특정 논문이나 모델은 목표가 아니라 교체
가능한 구현 레퍼런스다.

## Historical version 0 decision

The first experimental trainable baseline followed the permissively licensed
one-shot identity/attribute/AAD family exemplified by GHOST v1 (Apache-2.0). It
is retained as an optimization experiment, not as the selected final
architecture. The code in
`src/face_realistic/modeling/` is an independent PyTorch implementation and
does not copy, load, distill, or depend on GHOST or InSwapper checkpoints.

## Data flow

```text
source identity RGB -> trainable IdentityEncoder ---------+
                                                          |
target RGB -> 16x16 low-frequency bottleneck              |
           -> multi-scale AttributeEncoder ---------------+-> AAD Generator
                                                          |      |-> RGB
MediaPipe blendshapes + normalized head pose -> MotionEncoder ---+-> alpha
```

The target bottleneck intentionally removes high-frequency identity cues while
retaining coarse pose, exposure, and layout. Explicit motion values restore the
expression controls that a low-resolution target cannot preserve reliably.

## Version 0 limitations

- One identity verifies optimization only; it cannot prove disentanglement.
- No third-party identity or perceptual checkpoint is used in the first loss.
- The face mask is an approximate aligned oval until rights-cleared parsing
  labels or a trainable visibility head are available.
- Temporal training is deferred until the frame model is useful.

## Expansion gates

1. Overfit `person_01` and reconstruct held-out expressions.
2. Add at least two identities and verify that source identity changes output.
3. Train an identity encoder on rights-cleared synthetic identities.
4. Add eye, lip-aperture, gaze, occlusion, and temporal objectives.
5. Export the generator to ONNX and compare against the frozen InSwapper
   research baseline for quality, identity leakage, and runtime.
