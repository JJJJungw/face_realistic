# Third-party and model boundary

## Clean-room trainable model

`src/face_realistic/modeling/` is original project code inspired by published
one-shot face-swap concepts, particularly GHOST v1. GHOST v1 source code is
Apache-2.0 licensed. No GHOST or InSwapper source file or checkpoint is bundled
or loaded by the clean-room training path.

Reference:

- GHOST: Alexander Groshev et al., "GHOST—A New Face Swap Approach for Image
  and Video Domains," IEEE Access, 2022.
- Source repository: https://github.com/ai-forever/ghost

## Research-only comparison path

The existing `inswapper_128.onnx` and InsightFace pretrained models are not
part of the clean-room model. InsightFace states that its code is MIT licensed,
while its distributed pretrained models and associated training data are for
non-commercial research unless separately licensed. They remain isolated as a
benchmark and must not be used as teachers, initialization, or training labels.

## Data

Only identity images and synthetic renders whose training rights have been
verified may be used to produce project checkpoints. Repository code licenses
do not grant rights to third-party datasets, model weights, or generated media.

