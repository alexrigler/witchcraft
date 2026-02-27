mod quantized_t5;
#[cfg(feature = "burn-bench")]
mod burn_t5;

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::time::Instant;

fn load_tokenizer(assets: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let compressed = std::fs::read(assets.join("tokenizer.json.zst"))?;
    let bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(tokenizer)
}

/// Repeat text to reach at least `min_tokens` after tokenization.
fn make_input(tokenizer: &tokenizers::Tokenizer, base: &str, min_tokens: usize) -> Vec<u32> {
    let mut text = base.to_string();
    loop {
        let enc = tokenizer.encode(text.as_str(), true).unwrap();
        if enc.get_ids().len() >= min_tokens {
            return enc.get_ids()[..min_tokens].to_vec();
        }
        text.push(' ');
        text.push_str(base);
    }
}

fn bench_candle(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer, sizes: &[usize]) -> Result<()> {
    let device = Device::Cpu;

    let compressed = std::fs::read(assets.join("config.json.zst"))?;
    let cfg_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
    let config: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;

    let t0 = Instant::now();
    let compressed = std::fs::read(assets.join("xtr.gguf.zst"))?;
    let model_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        &model_bytes,
        &device,
    )?;
    let model = quantized_t5::T5EncoderModel::load(vb, &config)?;
    eprintln!("candle: model loaded in {:.0?}", t0.elapsed());

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    // Warmup
    let ids = make_input(tokenizer, base, 32);
    let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
    let _ = model.forward(&input)?;

    for &n in sizes {
        let ids = make_input(tokenizer, base, n);
        let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;

        // Warmup this size
        let _ = model.forward(&input)?;

        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            let out = model.forward(&input)?;
            let _ = out.dims3()?;
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[3];
        eprintln!(
            "candle:   {:>4} tokens -> median {:>7.1?}  ({:.0} tok/s)",
            n, median, n as f64 / median.as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(feature = "ov")]
fn bench_openvino(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer, sizes: &[usize]) -> Result<()> {
    use openvino::{Core, DeviceType, Shape};

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    // For each size, create a fresh OV model+infer request to avoid shape-change issues.
    // Retry on failure.
    for &n in sizes {
        let ids = make_input(tokenizer, base, n);
        let padded_len = n.next_power_of_two().max(64);

        let mut success = false;
        for attempt in 0..3 {
            let result: Result<()> = (|| {
                let mut core = Core::new().map_err(|e| anyhow::anyhow!("ov core: {e:?}"))?;
                let temp_dir = tempfile::tempdir()?;
                let xml_path = temp_dir.path().join("model.xml");
                {
                    let compressed = std::fs::read(assets.join("xtr-ov-int4.xml.zst"))?;
                    let xml_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
                    std::fs::write(&xml_path, xml_bytes)?;
                }
                let bin_path = assets.join("xtr-ov-int4.bin");

                let model = core.read_model_from_file(
                    xml_path.to_str().unwrap(),
                    bin_path.to_str().unwrap(),
                ).map_err(|e| anyhow::anyhow!("read model: {e:?}"))?;

                let mut compiled = core.compile_model(&model, DeviceType::CPU)
                    .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
                let mut infer = compiled.create_infer_request()
                    .map_err(|e| anyhow::anyhow!("infer request: {e:?}"))?;

                let mut run_once = |ids: &[u32]| -> Result<()> {
                    let mut padded: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
                    padded.resize(padded_len, 0);
                    let ov_shape = Shape::new(&[1i64, padded_len as i64])
                        .map_err(|e| anyhow::anyhow!("shape: {e:?}"))?;
                    let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
                        .map_err(|e| anyhow::anyhow!("tensor: {e:?}"))?;
                    ov_input.get_data_mut::<i64>()
                        .map_err(|e| anyhow::anyhow!("data: {e:?}"))?
                        .copy_from_slice(&padded);
                    infer.set_input_tensor(&ov_input)
                        .map_err(|e| anyhow::anyhow!("set input: {e:?}"))?;
                    infer.infer()
                        .map_err(|e| anyhow::anyhow!("infer: {e:?}"))?;
                    let _output = infer.get_output_tensor()
                        .map_err(|e| anyhow::anyhow!("output: {e:?}"))?;
                    Ok(())
                };

                // Warmup (3 iters, same shape)
                for _ in 0..3 {
                    run_once(&ids)?;
                }

                let mut times = Vec::new();
                for _ in 0..7 {
                    let t = Instant::now();
                    run_once(&ids)?;
                    times.push(t.elapsed());
                }
                times.sort();
                let median = times[3];
                eprintln!(
                    "openvino: {:>4} tokens (padded {:>4}) -> median {:>7.1?}  ({:.0} tok/s)",
                    n, padded_len, median, n as f64 / median.as_secs_f64()
                );
                Ok(())
            })();

            match result {
                Ok(()) => { success = true; break; }
                Err(e) => {
                    eprintln!("openvino: {} tokens attempt {} failed: {e}", n, attempt + 1);
                }
            }
        }
        if !success {
            eprintln!("openvino: {} tokens FAILED after 3 attempts", n);
        }
    }
    Ok(())
}

#[cfg(feature = "burn-bench")]
fn bench_burn(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer, sizes: &[usize]) -> Result<()> {
    let safetensors_path = assets.join("xtr-f32.safetensors");

    let t0 = Instant::now();
    let model = burn_t5::BurnT5Encoder::load(&safetensors_path);
    eprintln!("burn: model loaded in {:.0?}", t0.elapsed());

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    // Warmup
    let ids = make_input(tokenizer, base, 32);
    let _ = model.forward(&ids);

    for &n in sizes {
        let ids = make_input(tokenizer, base, n);

        // Warmup this size
        let _ = model.forward(&ids);

        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            let out = model.forward(&ids);
            let _shape = out.dims();
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[3];
        eprintln!(
            "burn:     {:>4} tokens -> median {:>7.1?}  ({:.0} tok/s)",
            n, median, n as f64 / median.as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(feature = "burn-bench")]
fn validate(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
    let device = Device::Cpu;
    let base = "Bananas are berries but strawberries are not.";
    let ids = make_input(tokenizer, base, 32);

    // Candle (Q4K dequantized to F32)
    let compressed = std::fs::read(assets.join("config.json.zst"))?;
    let cfg_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
    let config: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;
    let compressed = std::fs::read(assets.join("xtr.gguf.zst"))?;
    let model_bytes = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        &model_bytes, &device,
    )?;
    let candle_model = quantized_t5::T5EncoderModel::load(vb, &config)?;
    let candle_input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
    let candle_out = candle_model.forward(&candle_input)?;
    let candle_data: Vec<f32> = candle_out.flatten_all()?.to_vec1()?;

    // Print token IDs for cross-checking with PyTorch
    eprintln!("token ids: {:?}", &ids);

    // Burn (F32)
    let burn_model = burn_t5::BurnT5Encoder::load(&assets.join("xtr-f32.safetensors"));
    let burn_out = burn_model.forward(&ids);
    let burn_data: Vec<f32> = burn_out.into_data().to_vec().unwrap();

    assert_eq!(candle_data.len(), burn_data.len(),
        "output size mismatch: candle={} burn={}", candle_data.len(), burn_data.len());
    let n = candle_data.len();
    eprintln!("output size: {} (32 tokens x 128 dims)", n);

    // Cosine similarity
    let dot: f64 = candle_data.iter().zip(&burn_data).map(|(a, b)| *a as f64 * *b as f64).sum();
    let norm_a: f64 = candle_data.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = burn_data.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (norm_a * norm_b);

    // Max absolute error
    let max_abs: f32 = candle_data.iter().zip(&burn_data)
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean_abs: f64 = candle_data.iter().zip(&burn_data)
        .map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / n as f64;

    eprintln!("cosine similarity: {cosine:.6}");
    eprintln!("mean absolute error: {mean_abs:.6}");
    eprintln!("max absolute error:  {max_abs:.6}");

    // Print first 8 values from each
    eprintln!("\nfirst 8 values (token 0):");
    eprintln!("  candle: {:?}", &candle_data[..8]);
    eprintln!("  burn:   {:?}", &burn_data[..8]);

    if cosine > 0.95 {
        eprintln!("\nVALIDATION PASSED (cosine > 0.95)");
    } else {
        eprintln!("\nVALIDATION FAILED (cosine = {cosine:.4}, expected > 0.95)");
    }
    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    eprintln!("assets dir: {}", assets.display());

    let tokenizer = load_tokenizer(&assets)?;

    let sizes = vec![32, 64, 128, 256, 512];
    let only = std::env::args().nth(2);

    let only = only.as_deref();

    #[cfg(feature = "burn-bench")]
    if only == Some("validate") {
        eprintln!("\n=== Validating Burn vs Candle ===");
        return validate(&assets, &tokenizer);
    }

    if only.is_none() || only == Some("candle") {
        eprintln!("\n=== Candle (Q4K -> F32 on CPU) ===");
        if let Err(e) = bench_candle(&assets, &tokenizer, &sizes) {
            eprintln!("candle error: {e}");
        }
    }

    #[cfg(feature = "ov")]
    if only.is_none() || only == Some("ov") {
        eprintln!("\n=== OpenVINO (INT4) ===");
        if let Err(e) = bench_openvino(&assets, &tokenizer, &sizes) {
            eprintln!("openvino error: {e}");
        }
    }

    #[cfg(feature = "burn-bench")]
    if only.is_none() || only == Some("burn") {
        eprintln!("\n=== Burn (NdArray F32) ===");
        if let Err(e) = bench_burn(&assets, &tokenizer, &sizes) {
            eprintln!("burn error: {e}");
        }
    }

    Ok(())
}
