// T5 Encoder implementation using Apple BNNS
#![allow(dead_code)]

use crate::bnns::{ffi::*, types::*};
use crate::{embed_raw_asset, embed_zst_asset};
use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// Embed assets using the macro from assets.rs
// BNNS uses SafeTensors format (same as OpenVINO), not GGUF
embed_zst_asset!(pub CONFIG,    "config.json.zst");
embed_zst_asset!(pub TOKENIZER, "tokenizer.json.zst");
embed_raw_asset!(pub MODEL,     "xtr-f32.safetensors");

const D_MODEL: usize = 768;
const NUM_HEADS: usize = 12;
const D_KV: usize = 64; // d_model / num_heads
const D_FF: usize = 3072;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

/// Builder for T5 encoder model
pub struct T5ModelBuilder {
    config: Config,
}

impl T5ModelBuilder {
    pub fn load(assets: &PathBuf) -> Result<(Self, Tokenizer)> {
        // Load config
        let cfg_bytes = CONFIG.bytes(assets).map_err(|_| anyhow!("Failed to load config"))?;
        let config: Config = serde_json::from_slice(cfg_bytes)?;

        // Load tokenizer
        let tok_bytes = TOKENIZER.bytes(assets).map_err(|_| anyhow!("Failed to load tokenizer"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes).map_err(anyhow::Error::msg)?;

        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(&self, device: &Device, assets: &PathBuf) -> Result<T5EncoderModel> {
        // Load SafeTensors model
        #[cfg(feature = "embed-assets")]
        let vb = {
            let model_bytes = MODEL.bytes().to_vec();
            VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, device)?
        };

        #[cfg(not(feature = "embed-assets"))]
        let vb = {
            let model_path = MODEL.path(assets);
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? }
        };

        T5EncoderModel::new(vb, device)
    }
}

/// T5 Encoder using BNNS backend
pub struct T5EncoderModel {
    layers: Vec<T5EncoderLayer>,
    final_norm: LayerNorm,
    embedding_weights: Vec<f32>,
    vocab_size: usize,
    d_model: usize,
    device: Device,
}

struct T5EncoderLayer {
    // Self-attention
    attention: MultiHeadAttention,
    attn_layer_norm: LayerNorm,

    // Feed-forward network
    ffn_wi: BnnsFilter,  // Includes GELU activation
    ffn_wo: BnnsFilter,
    ffn_wi_weights: Vec<f32>,
    ffn_wo_weights: Vec<f32>,
    ffn_layer_norm: LayerNorm,
}

struct MultiHeadAttention {
    // Q/K/V projection weights (stored for BNNS descriptor)
    q_weights: Vec<f32>,
    k_weights: Vec<f32>,
    v_weights: Vec<f32>,
    o_weights: Vec<f32>,

    // Optional biases
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    o_bias: Vec<f32>,

    num_heads: usize,
    d_model: usize,
    d_kv: usize,

    // We'll create filters dynamically per-batch since BNNS MHA needs runtime descriptors
}

struct LayerNorm {
    filter: BnnsFilter,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    d_model: usize,
    epsilon: f32,
}

impl T5EncoderModel {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        log::info!("[BNNS] Creating T5 encoder...");

        // Load embedding weights
        let embedding_weights = vb
            .get((50358, D_MODEL), "shared.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Create encoder layers
        let num_layers = 12; // T5-base has 12 encoder layers
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let layer_vb = vb.pp(&format!("encoder.block.{i}"));
            layers.push(T5EncoderLayer::new(layer_vb)?);
        }

        // Final layer norm
        let final_norm_vb = vb.pp("encoder.final_layer_norm");
        let final_norm = LayerNorm::new(final_norm_vb, D_MODEL)?;

        log::info!("[BNNS] T5 encoder created with {} layers", num_layers);

        Ok(Self {
            layers,
            final_norm,
            embedding_weights,
            vocab_size: 50358,
            d_model: D_MODEL,
            device: device.clone(),
        })
    }

    /// Encode input token IDs to embeddings
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Extract token IDs from tensor
        let input_ids = input.squeeze(0)?.to_vec1::<i64>()?;
        let seq_len = input_ids.len();
        log::debug!("[BNNS] Encoding {} tokens", seq_len);

        // Embedding lookup
        let mut hidden = self.embed(&input_ids)?;

        // Run through encoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, seq_len)?;
            if i % 4 == 0 {
                log::debug!("[BNNS] Completed layer {}/{}", i + 1, self.layers.len());
            }
        }

        // Final layer norm
        hidden = self.final_norm.forward(hidden, seq_len)?;

        // Convert back to candle Tensor
        self.to_tensor(hidden, seq_len)
    }

    fn embed(&self, input_ids: &[i64]) -> Result<BnnsTensor> {
        let seq_len = input_ids.len();
        let mut output = BnnsTensor::new(vec![seq_len, self.d_model]);

        for (i, &token_id) in input_ids.iter().enumerate() {
            let token_id = token_id as usize;
            if token_id >= self.vocab_size {
                return Err(anyhow!("Token ID {} out of range", token_id));
            }

            let embedding_start = token_id * self.d_model;
            let embedding_end = embedding_start + self.d_model;
            let embedding = &self.embedding_weights[embedding_start..embedding_end];

            let output_start = i * self.d_model;
            let output_end = output_start + self.d_model;
            output.data[output_start..output_end].copy_from_slice(embedding);
        }

        Ok(output)
    }

    fn to_tensor(&self, bnns_tensor: BnnsTensor, seq_len: usize) -> Result<Tensor> {
        // Convert BnnsTensor back to candle Tensor
        let data = bnns_tensor.data;
        let tensor = Tensor::from_vec(data, (seq_len, self.d_model), &self.device)?;
        Ok(tensor)
    }
}

impl T5EncoderLayer {
    fn new(vb: VarBuilder) -> Result<Self> {
        // Load attention weights
        let attn_vb = vb.pp("layer.0.SelfAttention");
        let attention = MultiHeadAttention::new(attn_vb)?;

        // Attention layer norm
        let attn_norm_vb = vb.pp("layer.0.layer_norm");
        let attn_layer_norm = LayerNorm::new(attn_norm_vb, D_MODEL)?;

        // FFN weights
        let ffn_vb = vb.pp("layer.1.DenseReluDense");

        let ffn_wi_weights = ffn_vb
            .get((D_FF, D_MODEL), "wi.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let ffn_wo_weights = ffn_vb
            .get((D_MODEL, D_FF), "wo.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Create FFN filters (we'll create them once per layer, reuse for all batches)
        // For now, store weights and create filters in forward pass
        // (BNNS filters need runtime descriptors)

        let ffn_wi = Self::create_dummy_filter();
        let ffn_wo = Self::create_dummy_filter();

        // FFN layer norm
        let ffn_norm_vb = vb.pp("layer.1.layer_norm");
        let ffn_layer_norm = LayerNorm::new(ffn_norm_vb, D_MODEL)?;

        Ok(Self {
            attention,
            attn_layer_norm,
            ffn_wi,
            ffn_wo,
            ffn_wi_weights,
            ffn_wo_weights,
            ffn_layer_norm,
        })
    }

    fn create_dummy_filter() -> BnnsFilter {
        // Placeholder - we'll create actual filters in forward pass
        // For now, return a null handle that we'll replace
        unsafe { std::mem::zeroed() }
    }

    fn forward(&self, mut hidden: BnnsTensor, seq_len: usize) -> Result<BnnsTensor> {
        // Save residual
        let residual = hidden.data.clone();

        // Self-attention
        hidden = self.attention.forward(hidden, seq_len)?;

        // Residual connection
        for i in 0..hidden.data.len() {
            hidden.data[i] += residual[i];
        }

        // Layer norm
        hidden = self.attn_layer_norm.forward(hidden, seq_len)?;

        // FFN with residual
        let residual = hidden.data.clone();

        // FFN forward pass
        hidden = self.ffn_forward(hidden, seq_len)?;

        // Residual connection
        for i in 0..hidden.data.len() {
            hidden.data[i] += residual[i];
        }

        // Layer norm
        hidden = self.ffn_layer_norm.forward(hidden, seq_len)?;

        Ok(hidden)
    }

    fn ffn_forward(&self, input: BnnsTensor, seq_len: usize) -> Result<BnnsTensor> {
        // FFN: wi (with GELU) -> wo
        // For now, implement manually (TODO: use BNNS filters)

        let mut intermediate = BnnsTensor::new(vec![seq_len, D_FF]);

        // wi projection + GELU
        for i in 0..seq_len {
            for j in 0..D_FF {
                let mut sum = 0.0;
                for k in 0..D_MODEL {
                    let input_val = input.data[i * D_MODEL + k];
                    let weight_val = self.ffn_wi_weights[j * D_MODEL + k];
                    sum += input_val * weight_val;
                }
                // GELU activation
                intermediate.data[i * D_FF + j] = gelu(sum);
            }
        }

        // wo projection
        let mut output = BnnsTensor::new(vec![seq_len, D_MODEL]);
        for i in 0..seq_len {
            for j in 0..D_MODEL {
                let mut sum = 0.0;
                for k in 0..D_FF {
                    let input_val = intermediate.data[i * D_FF + k];
                    let weight_val = self.ffn_wo_weights[j * D_FF + k];
                    sum += input_val * weight_val;
                }
                output.data[i * D_MODEL + j] = sum;
            }
        }

        Ok(output)
    }
}

impl MultiHeadAttention {
    fn new(vb: VarBuilder) -> Result<Self> {
        // Load Q/K/V/O projection weights
        let q_weights = vb
            .get((D_MODEL, D_MODEL), "q.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let k_weights = vb
            .get((D_MODEL, D_KV), "k.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let v_weights = vb
            .get((D_MODEL, D_KV), "v.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let o_weights = vb
            .get((D_MODEL, D_MODEL), "o.weight")?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        Ok(Self {
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            q_bias: vec![],
            k_bias: vec![],
            v_bias: vec![],
            o_bias: vec![],
            num_heads: NUM_HEADS,
            d_model: D_MODEL,
            d_kv: D_KV,
        })
    }

    fn forward(&self, input: BnnsTensor, _seq_len: usize) -> Result<BnnsTensor> {
        // TODO: Use BNNS BNNSFilterCreateLayerMultiheadAttention
        // For now, implement manually

        // Simplified attention: just return input (placeholder)
        log::warn!("[BNNS] Attention not yet implemented, returning identity");
        Ok(input)
    }
}

impl LayerNorm {
    fn new(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let gamma = vb
            .get(d_model, "weight")?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;

        // T5 uses RMSNorm (no beta), but we'll handle it
        let beta = vec![0.0; d_model];

        // Create BNNS layer norm filter
        // For now, create dummy filter
        let filter = unsafe { std::mem::zeroed() };

        Ok(Self {
            filter,
            gamma,
            beta,
            d_model,
            epsilon: 1e-6,
        })
    }

    fn forward(&self, mut input: BnnsTensor, seq_len: usize) -> Result<BnnsTensor> {
        // RMS normalization (T5 uses RMSNorm variant)
        for i in 0..seq_len {
            let start = i * self.d_model;
            let end = start + self.d_model;
            let slice = &mut input.data[start..end];

            // Compute RMS
            let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sum_sq / self.d_model as f32 + self.epsilon).sqrt();

            // Normalize and scale
            for (j, val) in slice.iter_mut().enumerate() {
                *val = (*val / rms) * self.gamma[j];
            }
        }

        Ok(input)
    }
}

// GELU activation (exact)
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

// === Public API matching other T5 implementations ===

impl T5EncoderModel {
    pub fn from_safetensors(_path: &std::path::Path, _device: &Device) -> Result<Self> {
        // For now, expect weights to be loaded via VarBuilder
        Err(anyhow!("Use T5EncoderModel::new with VarBuilder instead"))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
