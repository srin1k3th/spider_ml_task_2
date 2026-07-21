## Prediction Quality

The Transformer outperformed the LSTM on every metric in this comparison.

| Metric | LSTM | Transformer | Better |
|---|---|---|---|
| MAE | 1.4627 | 1.4152 | Transformer |
| RMSE | 1.9576 | 1.9130 | Transformer |
| R² | 0.9369 | 0.9398 | Transformer |
| Correlation | 0.9683 | 0.9698 | Transformer |
| Best window MAE | 0.1008 | 0.0826 | Transformer |
| Median window MAE | 1.2510 | 1.2071 | Transformer |
| Worst window MAE | 9.9659 | 7.6908 | Transformer |

Both models achieved an R² above 0.93. This means both models explained more than 93% of the variance in the test data. The difference between them is small but consistent. The Transformer had a lower MAE by about 0.05°C and a lower RMSE by about 0.04°C.

The most notable difference is in the worst-case window. The LSTM had a worst MAE of 9.97°C. The Transformer had a worst MAE of 7.69°C. This shows that the Transformer handled difficult samples better. It produced fewer extreme errors.

The error threshold table confirms this pattern:

| Threshold | LSTM | Transformer |
|---|---|---|
| > 1°C | 67,836 | 65,100 |
| > 2°C | 32,435 | 31,089 |
| > 3°C | 14,604 | 13,962 |
| > 4°C | 6,293 | 5,953 |
| > 5°C | 2,633 | 2,570 |
| > 6°C | 1,084 | 1,123 |

The Transformer produced fewer errors above each threshold up to 5°C. At the > 6°C threshold, the LSTM had slightly fewer extreme errors (1,084 vs 1,123). This suggests that the Transformer reduces moderate errors more effectively, but a small number of its worst predictions can be more severe.

## Long-Range Dependency Handling

The task requires the model to take 72 hours of input and predict the next 12 hours. This means the model must connect patterns across a 72-step window.

The LSTM processes these 72 steps one at a time. It must compress all relevant context into a fixed-size hidden state. By the time it reaches step 72, early information from step 1 has passed through 71 sequential gates. Some of that information will be weakened or lost.

The Transformer accesses all 72 positions in parallel through self-attention. Each position can attend directly to every other position. The path length between any two steps is always one. This direct access explains why the Transformer captured patterns more accurately, especially in difficult windows where long-range context matters most.

The worst-case MAE difference (9.97°C vs 7.69°C) supports this theory. Difficult windows likely contain unusual patterns that require the model to use context from many time steps back. The Transformer preserved that context better than the LSTM.

## Runtime

The LSTM took approximately 17 minutes to train for 25 epochs. The Transformer took approximately 2 minutes for the same 25 epochs. The Transformer was roughly 8.5 times faster.

This difference comes from how each architecture processes the input sequence:

- **LSTM**: It must process each time step in order. The output of step *t* depends on the output of step *t−1*. This sequential dependency prevents the GPU from computing all steps at the same time. The GPU cores sit idle while they wait for each step to finish.
- **Transformer**: It processes all time steps in parallel. The self-attention mechanism computes relationships between all positions in a single matrix operation. This fully uses the parallel compute capacity of the GPU.

The 8.5x speed advantage is significant. It means the Transformer can train on larger datasets or run more experiments in the same amount of time.

## Memory Usage

No direct memory measurements were taken during this comparison. The following analysis is based on the architectural properties of each model.

**LSTM memory characteristics:**
- Memory use grows linearly with sequence length. At each step, the model stores one hidden state and one cell state. Both have a fixed size equal to the hidden dimension (64 in this implementation).
- Backpropagation through time (BPTT) requires the model to store intermediate states for all 72 time steps. This is the main source of memory use during training.
- Total parameter count is relatively small. An LSTM layer with 64 hidden units and multi-feature input has a modest number of weights.

**Transformer memory characteristics:**
- The self-attention mechanism computes a 72×72 attention matrix for each head and each sample in the batch. Memory use grows with the square of the sequence length.
- With 4 heads and 3 layers, the model stores 12 attention matrices per sample during the forward pass.
- The feed-forward layers (d_model × 4 × d_model) also add to memory use.
- For a 72-step sequence, the quadratic cost is still manageable. The Transformer memory overhead becomes a problem only for much longer sequences (thousands of steps).

In this task, both models likely used similar amounts of GPU memory. The sequence length of 72 is short enough that the quadratic attention cost does not dominate.

## Training Stability

Both models used the same training setup: 25 epochs, Adam optimizer, learning rate warmup for 5 epochs, ReduceLROnPlateau scheduler, weight decay of 5e-4, and gradient clipping.

**LSTM training behavior:**
- The LSTM loss decreased smoothly from epoch 1 to epoch 25.
- Training loss went from 0.3093 to 0.0501. Validation loss went from 0.1376 to 0.0573.
- The validation loss showed very little fluctuation after epoch 7. It decreased steadily and the gap between training and validation loss stayed small.
- Best validation loss: 0.0573.

**Transformer training behavior:**
- The Transformer loss also decreased, but the validation loss showed more fluctuation.
- Training loss went from 0.2744 to 0.0538. Validation loss went from 0.1139 to 0.0610.
- The validation loss spiked at epoch 8 (0.0753) before recovering. It continued to fluctuate between 0.06 and 0.068 for the remaining epochs.
- Best validation loss: 0.0607 (at epoch 17).

The LSTM was more stable during training. Its loss curve was smoother and it converged more predictably. The Transformer showed occasional validation spikes. This is a known behavior of attention-based models. The attention weights can shift suddenly during training, which causes temporary increases in the loss. The warmup schedule and learning rate reduction helped control this, but did not eliminate it entirely.

Despite the less stable training, the Transformer achieved better test metrics. This suggests that the fluctuations did not harm the final model quality.

## Strengths and Limitations of Each Architecture

### LSTM

**Strengths:**
- Smooth, predictable training. The loss curves are stable and easy to monitor.
- Lower parameter count. The model is smaller and simpler to deploy.
- Linear memory scaling. Memory use stays proportional to sequence length, which makes it safe for very long sequences.
- No positional encoding needed. The sequential processing gives the model implicit order information.

**Limitations:**
- Slow training. Sequential processing prevents full GPU utilization. The 17-minute training time was 8.5 times longer than the Transformer.
- Weaker long-range context. The fixed-size hidden state cannot hold all information from a 72-step window equally well. This caused higher worst-case errors.
- Limited scalability. Adding more layers or increasing the hidden size does not improve LSTM performance as effectively as scaling a Transformer.

### Transformer

**Strengths:**
- Better prediction quality. It won on every metric in this comparison.
- Stronger worst-case performance. The worst window MAE was 23% lower than the LSTM (7.69°C vs 9.97°C).
- Fast training. Parallel processing completed training in 2 minutes.
- Effective long-range modeling. Direct attention connections let the model use context from any part of the 72-step input.

**Limitations:**
- Less stable training. Validation loss fluctuated more than the LSTM. This requires more careful hyperparameter tuning.
- Quadratic memory scaling. For much longer sequences, the attention matrices would consume significantly more memory.
- Requires positional encoding. The model has no built-in sense of order. It depends on the sinusoidal positional encoding to understand time step positions.
- More parameters. The query, key, value projections and multi-head structure add complexity.

### Summary

For this 72→12 hour temperature forecasting task, the Transformer was the better choice. It trained faster, produced better predictions, and handled difficult samples more robustly. The LSTM offered more stable training but could not match the Transformer on accuracy or speed. For tasks with much longer sequences and tight memory constraints, the LSTM could still be the more practical option.
