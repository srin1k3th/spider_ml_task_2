## Why residual connections help optimization

Deep plain CNNs suffer from the degradation problem, which is the fact that past a certain depth, training accuracy itself gets worse, not just validation accuracy. This isn't overfitting, but rather an optimization failure. Each layer in a plain stack has to learn a full mapping H(x), and if the ideal mapping for a given block is close to identity, a stack of nonlinear conv/ReLU layers is surprisingly bad at learning to "do nothing."

Residual blocks fix this. Instead of learning H(x) directly, the block learns F(x) = H(x) - x, and the true output is computed as F(x) + x. If identity really is the ideal mapping, the block just needs F(x) tending to 0, which is trivial for gradient descent, instead of learning an exact identity through nonlinear layers, which is hard. This doesn't reduce expressive power, as F(x) can still represent anything; it just makes the "easy case" easy to learn.

## How skip connections improve gradient flow

During backprop, `out = F(x) + x` gives gradients two paths back to `x`: through F(x) (multiplied by that branch's local derivatives, same as a plain network) and through the identity branch, whose local derivative at the addition node is exactly 1, regardless of how many layers deep F(x) goes.

In a plain network, backprop to an early layer multiplies local derivatives across every layer, and if those derivatives are consistently below 1 (common with conv/BN/ReLU), the product shrinks toward zero across depth. This is the vanishing gradient problem. With residual connections, every block's identity branch adds +1 to that multiplicative chain, so even in the worst case (F'(x) collapsing to 0 at every layer), the gradient reaching an early layer is still at least 1, not exponentially decayed. The `+1` acts as a floor that prevents the systematic collapse toward zero.

## The effect of network depth on performance

Depth increases representational capacity, but two separate costs work against "deeper is always better":

1. **Degradation problem** (addressed above) : Mostly resolved by residual connections, so a ResNet-style network doesn't suffer training-accuracy collapse at depths where a plain network might.
2. **Capacity vs. dataset size** : More layers means more parameters, which raises overfitting risk relative to a fixed-size dataset. CIFAR-10's 50k training images (40k after our validation split) is small by deep learning standards. A deeper/wider network than necessary risks memorizing training-set specifics rather than learning generalizable features.


## Challenges encountered during implementation

The main challenge I faced here was about runtime and some syntax errors.
**Small batch size (20) causing painfully slow training on a T4 GPU.** Small batches underuse GPU parallelism due to per-batch launch limits.

On the actual ResNet code itself, the main design challenge was making the downsample/projection path trigger only exactly when needed (`stride != 1 or in_channels != out_channels`), so dimension mismatches between the main path and the skip path are resolved properly.
