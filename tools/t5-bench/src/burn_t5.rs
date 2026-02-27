use burn::prelude::*;
use burn::tensor::activation;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

/// Load a named F32 tensor from safetensors into a Burn tensor.
fn load_tensor<const D: usize>(
    tensors: &safetensors::SafeTensors,
    name: &str,
) -> Tensor<B, D> {
    let view = tensors.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(
        view.dtype(),
        safetensors::Dtype::F32,
        "{name}: expected F32, got {:?}",
        view.dtype()
    );
    let data: &[u8] = view.data();
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape: Vec<usize> = view.shape().to_vec();
    Tensor::from_data(TensorData::new(floats, shape), &Default::default())
}

fn rms_norm(xs: Tensor<B, 3>, weight: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let variance = xs.clone().powf_scalar(2.0).mean_dim(2);
    let xs = xs / (variance + eps).sqrt();
    xs * weight.clone().unsqueeze()
}

fn softmax(xs: Tensor<B, 4>) -> Tensor<B, 4> {
    activation::softmax(xs, 3)
}

fn gelu(xs: Tensor<B, 3>) -> Tensor<B, 3> {
    activation::gelu(xs)
}

/// xs @ weight^T, broadcasting weight [out, in] -> [1, in, out] for batched matmul.
fn linear(xs: Tensor<B, 3>, weight: &Tensor<B, 2>) -> Tensor<B, 3> {
    xs.matmul(weight.clone().transpose().unsqueeze_dim(0))
}

struct T5Block {
    attn_norm: Tensor<B, 1>,
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    o: Tensor<B, 2>,
    relative_attention_bias: Option<Tensor<B, 2>>,
    ff_norm: Tensor<B, 1>,
    wi_0: Tensor<B, 2>,
    wi_1: Tensor<B, 2>,
    wo: Tensor<B, 2>,
}

pub struct BurnT5Encoder {
    embed_tokens: Tensor<B, 2>,
    blocks: Vec<T5Block>,
    final_norm: Tensor<B, 1>,
    projection: Tensor<B, 2>,
    num_heads: usize,
    d_kv: usize,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
}

impl BurnT5Encoder {
    pub fn load(path: &std::path::Path) -> Self {
        let bytes = std::fs::read(path).expect("read safetensors");
        let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");

        let embed_tokens = load_tensor(&tensors, "encoder.embed_tokens.weight");
        let final_norm = load_tensor(&tensors, "encoder.final_layer_norm.weight");
        let projection = load_tensor(&tensors, "linear.weight");

        let blocks: Vec<T5Block> = (0..12)
            .map(|i| {
                let p = format!("encoder.block.{i}");
                let attn = format!("{p}.layer.0.SelfAttention");
                let ff = format!("{p}.layer.1.DenseReluDense");
                T5Block {
                    attn_norm: load_tensor(&tensors, &format!("{p}.layer.0.layer_norm.weight")),
                    q: load_tensor(&tensors, &format!("{attn}.q.weight")),
                    k: load_tensor(&tensors, &format!("{attn}.k.weight")),
                    v: load_tensor(&tensors, &format!("{attn}.v.weight")),
                    o: load_tensor(&tensors, &format!("{attn}.o.weight")),
                    relative_attention_bias: if i == 0 {
                        Some(load_tensor(
                            &tensors,
                            &format!("{attn}.relative_attention_bias.weight"),
                        ))
                    } else {
                        None
                    },
                    ff_norm: load_tensor(&tensors, &format!("{p}.layer.1.layer_norm.weight")),
                    wi_0: load_tensor(&tensors, &format!("{ff}.wi_0.weight")),
                    wi_1: load_tensor(&tensors, &format!("{ff}.wi_1.weight")),
                    wo: load_tensor(&tensors, &format!("{ff}.wo.weight")),
                }
            })
            .collect();

        Self {
            embed_tokens,
            blocks,
            final_norm,
            projection,
            num_heads: 12,
            d_kv: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
        }
    }

    /// Compute T5 relative position bias buckets (same logic as HuggingFace T5).
    fn compute_position_bias(&self, seq_len: usize) -> Tensor<B, 4> {
        let num_buckets = self.relative_attention_num_buckets as u32;
        let max_distance = self.relative_attention_max_distance as u32;
        let half = num_buckets / 2;
        let max_exact = half / 2;

        let mut buckets = vec![0u32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let relative_position = j as i64 - i as i64;
                let bucket = if relative_position > 0 {
                    let rel = relative_position as u32;
                    if rel < max_exact {
                        rel + half
                    } else {
                        let b = (rel as f32 / max_exact as f32).ln()
                            / (max_distance as f32 / max_exact as f32).ln()
                            * (half - max_exact) as f32;
                        u32::min(max_exact + half + b as u32, num_buckets - 1)
                    }
                } else {
                    let rel = (-relative_position) as u32;
                    if rel < max_exact {
                        rel
                    } else {
                        let b = (rel as f32 / max_exact as f32).ln()
                            / (max_distance as f32 / max_exact as f32).ln()
                            * (half - max_exact) as f32;
                        u32::min(max_exact + b as u32, half - 1)
                    }
                };
                buckets[i * seq_len + j] = bucket;
            }
        }

        // Gather from relative_attention_bias [32, 12] using bucket indices
        let bias_weight = self.blocks[0]
            .relative_attention_bias
            .as_ref()
            .unwrap();
        // bias_weight shape: [num_buckets=32, num_heads=12]
        // We need to index into it with our bucket indices to get [seq_len, seq_len, num_heads]
        // then permute to [1, num_heads, seq_len, seq_len]

        // Flatten bucket indices and gather
        let bias_data: Vec<f32> = burn::tensor::TensorData::from(bias_weight.clone().into_data())
            .to_vec()
            .unwrap();
        let num_heads = self.num_heads;

        let mut result = vec![0.0f32; seq_len * seq_len * num_heads];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let bucket = buckets[i * seq_len + j] as usize;
                for h in 0..num_heads {
                    result[i * seq_len * num_heads + j * num_heads + h] =
                        bias_data[bucket * num_heads + h];
                }
            }
        }

        // Shape [seq_len, seq_len, num_heads] -> permute to [num_heads, seq_len, seq_len] -> unsqueeze batch
        let t: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(result, [seq_len, seq_len, num_heads]), &Default::default());
        t.permute([2, 0, 1]).unsqueeze_dim(0)
    }

    fn self_attention(
        &self,
        block: &T5Block,
        xs: Tensor<B, 3>,
        position_bias: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, seq_len, _d_model] = xs.dims();
        let n_heads = self.num_heads;
        let d_kv = self.d_kv;

        // Q, K, V projections: [b, seq, d_model] x [d_model, inner_dim]^T = [b, seq, inner_dim]
        let q = linear(xs.clone(), &block.q);
        let k = linear(xs.clone(), &block.k);
        let v = linear(xs, &block.v);

        // Reshape to [b, seq, n_heads, d_kv] then transpose to [b, n_heads, seq, d_kv]
        let q = q.reshape([b, seq_len, n_heads, d_kv]).swap_dims(1, 2);
        let k = k.reshape([b, seq_len, n_heads, d_kv]).swap_dims(1, 2);
        let v = v.reshape([b, seq_len, n_heads, d_kv]).swap_dims(1, 2);

        // Attention scores: [b, n_heads, seq, seq]
        let scores = q.matmul(k.transpose());
        let scores = scores + position_bias.clone();
        let attn_weights = softmax(scores);

        // [b, n_heads, seq, d_kv]
        let attn_output = attn_weights.matmul(v);

        // [b, seq, n_heads, d_kv] -> [b, seq, inner_dim]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([b, seq_len, n_heads * d_kv]);

        // Output projection
        linear(attn_output, &block.o)
    }

    fn feed_forward(&self, block: &T5Block, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_gelu = gelu(linear(xs.clone(), &block.wi_0));
        let hidden_linear = linear(xs, &block.wi_1);
        let hidden = hidden_gelu * hidden_linear;
        linear(hidden, &block.wo)
    }

    pub fn forward(&self, input_ids: &[u32]) -> Tensor<B, 3> {
        let seq_len = input_ids.len();
        let device = Default::default();

        // Embedding lookup
        let indices: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::from(input_ids), &device);
        let hidden: Tensor<B, 3> = self.embed_tokens.clone().select(0, indices).unsqueeze_dim(0);

        let position_bias = self.compute_position_bias(seq_len);

        let mut xs = hidden;
        let eps = 1e-6;
        for block in &self.blocks {
            // Self-attention with pre-norm
            let normed = rms_norm(xs.clone(), &block.attn_norm, eps);
            let attn_out = self.self_attention(block, normed, &position_bias);
            xs = xs + attn_out;

            // Feed-forward with pre-norm
            let normed = rms_norm(xs.clone(), &block.ff_norm, eps);
            let ff_out = self.feed_forward(block, normed);
            xs = xs + ff_out;
        }

        // Final layer norm
        xs = rms_norm(xs, &self.final_norm, eps);

        // Final projection [b, seq, 768] x [768, 128]^T = [b, seq, 128]
        linear(xs, &self.projection)
    }
}
