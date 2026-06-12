# 02 — Classifiers: five ways to label a digit

Five image classifiers, one task (MNIST `classification` mode, see `docs/01-data.md`),
one training loop (`SupervisedTrainer` in `core/runtime.py`: Adam + cross-entropy +
accuracy). Because data and loop are fixed, the only variable is the architecture —
which is the whole point: you can read each model file in isolation and compare
accuracy per parameter directly in a check sweep.

## classifier — plain CNN

The baseline. Two `ConvBNReLU` blocks (`core/blocks/cnn.py`: Conv3x3 → BatchNorm →
ReLU), each followed by 2×2 max-pool (28→14→7), then flatten →
`Linear(64·7·7 → 64)` → ReLU → `Linear(64 → 10)`. Convolution gives translation
equivariance and weight sharing; pooling buys growing receptive fields. Everything
after this chapter is a refinement of this recipe.
Code: `src/mini_networks/models/classifier/model.py` (`SmallCNN`).

## resnet — residual connections

Deep plain CNNs degrade because each layer must learn a full mapping. ResNet's fix:
learn a *residual* `F(x)` and output `x + F(x)`, so identity is the default and
gradients flow through the skip. `BasicBlock` here is faithful: conv-BN, conv-BN, add
shortcut, ReLU after the add; a 1×1-conv + BN shortcut when stride or width changes.
`MiniResNet` = stem conv + 3 blocks (32 → 64 → 128 channels, strides 1/2/2) + global
average pool + linear. Simplified vs the paper: 3 blocks instead of 4 multi-block
stages, no 7×7 stem or initial max-pool (28×28 input doesn't need them).
Code: `src/mini_networks/models/resnet/model.py` (`MiniResNet`).

## vit — patches + attention

Vision Transformer: cut the image into patches, embed each as a token, and let
self-attention relate every patch to every other — global receptive field at layer 1,
no convolutional inductive bias. `MiniViT` patch-embeds with a strided
`Conv2d(1, 64, kernel=stride=4)` (49 patches), prepends a learnable CLS token, adds a
learned positional embedding, runs `nn.TransformerEncoder` (4 layers, 4 heads,
d_model 64, FFN 128, dropout 0.1), and classifies from the normed CLS token.
Simplified: stock PyTorch encoder layers rather than the paper's pre-norm blocks, no
augmentation/regularization stack, no pretraining — which is why its M-tier bar is
deliberately lower (0.75): ViTs are data-hungry, and that weakness is itself the
lesson. Code: `src/mini_networks/models/vit/model.py` (`MiniViT`).

## mobilenet — depthwise separable convolutions

MobileNet's idea: factor a standard convolution into a depthwise 3×3 (one filter per
channel, spatial mixing only) plus a pointwise 1×1 (channel mixing only) — roughly
8–9× fewer multiply-adds at similar accuracy. `DepthwiseSeparable`
(`core/blocks/cnn.py`) is `dw 3×3 (groups=in_ch) → pw 1×1 → BN → ReLU`.
`TinyMobileNet` = stem conv + 2 such blocks (stride 2 each) + pool + linear, with a
`width_mult` knob like the paper's width multiplier. Simplified: 2 blocks instead of
13, and one BN+ReLU after the pointwise conv instead of the paper's BN+ReLU after
*both* the depthwise and pointwise stages.
Code: `src/mini_networks/models/mobilenet/model.py` (`TinyMobileNet`).

## convnext — a CNN modernized to match transformers

ConvNeXt asks: how much of ViT's edge is the architecture recipe rather than
attention? Its block "modernizes" the ResNet block with transformer habits:
depthwise conv for spatial mixing (attention's role), LayerNorm instead of BatchNorm,
an inverted bottleneck (1×1 expand ×4 → GELU → 1×1 project, exactly a transformer
FFN), one activation per block, and a residual add. `ConvNeXtBlock` here implements
that recipe — including the channels-last permute that LayerNorm needs —
with a 3×3 depthwise kernel instead of the paper's 7×7 (28×28 inputs). Also dropped:
layer scale, stochastic depth, and the 4×4 patchify stem (we keep a `ConvBNReLU`
stem). `TinyConvNeXt` = stem + 3 stages (32 → 64 → 128) with strided-conv
downsampling between them + pool + linear.
Code: `src/mini_networks/models/convnext/model.py` (`TinyConvNeXt`).

## Quality bars

From `core/evalspec.py`, metric `accuracy`:

| Model | M tier | L tier |
|---|---|---|
| classifier, resnet | 0.85 | 0.95 |
| mobilenet, convnext | 0.80 | 0.93 |
| vit | 0.75 | 0.90 |

Run them: `uv run python main.py train --model resnet --training_tier M`, or gate all
five with `uv run python main.py sweep --check --models classifier,resnet,vit,mobilenet,convnext --skip-compositions`.

## Latest results

<!-- results:start items=classifier,resnet,vit,mobilenet,convnext -->

_Latest sweep: tier S on cpu_

| Item | Status | Metric | Value | Threshold |
|---|---|---|---|---|
| classifier | pass | accuracy | 0.0000 | n/a |
| resnet | pass | accuracy | 0.1250 | n/a |
| vit | pass | accuracy | 0.0625 | n/a |
| mobilenet | pass | accuracy | 0.0625 | n/a |
| convnext | pass | accuracy | 0.0625 | n/a |

<!-- results:end -->
