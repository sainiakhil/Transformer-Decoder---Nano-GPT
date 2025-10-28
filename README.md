# Tiny Shakespeare — Transformer Decoder (PyTorch)

A minimal transformer-decoder implementation in PyTorch trained on the Tiny Shakespeare dataset. This project demonstrates a simple autoregressive character-level language model (transformer decoder blocks, multi-head self-attention, and a small feed-forward network).
---

## Quick start

1. Install dependencies (recommended in a virtualenv):

```bash
python -m pip install --upgrade pip
pip install torch tqdm
```

2. Place the dataset file in the project root and name it `Text Dataset.txt`.

3. Run the training script:

```bash
python Self Attention Decoder.py
```

> The script as-is uses the hyperparameters at the top of the file (e.g. `batch_size`, `block_size`, `n_embd`, `n_layers`). Edit those values in the file before running if you want different settings.

---

## Expected files & outputs

* Model prints training & validation loss during training.
* After training finishes, the script runs `model.generate(...)` and prints generated text to stdout.

---

## Important notes & recommended fixes (must-read)

The provided script is a great starting point but contains a few practical issues and improvements you should apply before training on GPU:

1. **Move batches to the correct device**

   * `get_batch()` returns CPU tensors. Before calling the model, call:

     ```py
     xb, yb = xb.to(device), yb.to(device)
     ```

     inside the training loop so inputs/targets and model parameters are on the same device.

2. **Position embeddings device & shape**

   * Replace the line that builds positional embeddings with an index on the correct device and add batch dimension if needed:

     ```py
     pos_idx = torch.arange(T, device=idx.device)
     pos_emb = self.position_embedding_table(pos_idx)[None, :, :]
     ```

3. **Attention scaling factor**

   * When computing attention scores, scale by the head dimension (not `C`):

     ```py
     d_k = q.size(-1)
     wei = (q @ k.transpose(-2, -1)) * (d_k ** -0.5)
     ```

4. **Move `tril` buffer & mask handling**

   * The existing `tril` buffer is fine, but ensure masking uses the current sequence length `T`.

5. **Memory and stability**

   * The default hyperparameters (e.g. `batch_size=64`, `block_size=256`, `n_embd=384`, `n_layers=6`) can be large for small GPUs and may cause OOMs. To reduce memory:

     * Lower `batch_size` (e.g. `8` or `4`).
     * Reduce `n_embd`, `n_layers`, or `block_size`.
     * Use `torch.cuda.amp` for mixed-precision training.
     * Consider gradient checkpointing (`torch.utils.checkpoint`) or smaller head counts.

6. **Model/device consistency**

   * After `model = BigramLanguageModel()` call, you move the model to device with `m = model.to(device)` — use `model.to(device)` (or use `m` consistently). Ensure optimizer uses parameters of the moved model.

7. **Minor improvements**

   * Use `nn.GELU()` instead of `ReLU()` in feed-forward for smoother training (Transformer practice).
   * Use `nn.LayerNorm` in the expected places and ensure consistent ordering.

---

## How to generate text

* After training you can generate text with:

```py
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(decode(out[0].tolist()))
```

Make sure the `generate()` method is given a tensor on the same device the model lives on.

---
