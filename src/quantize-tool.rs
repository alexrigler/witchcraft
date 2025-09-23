use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result};
use std::io::{Seek, SeekFrom, Write};

fn main() -> anyhow::Result<()> {
    let in_file = "xtr.safetensors";
    let out_file = "assets/xtr.gguf.zst";
    let mut tmp = tempfile::tempfile()?;
    let tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
    println!("tensors: {}", tensors.len());

    let dtype = GgmlDType::Q4K;
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {dtype:?} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut tmp, &[], &qtensors)?;
    tmp.flush()?;
    tmp.seek(SeekFrom::Start(0))?;

    let raw_out = std::fs::File::create(out_file)?;
    let mut enc = zstd::Encoder::new(raw_out, 19)?;
    std::io::copy(&mut tmp, &mut enc)?;
    enc.finish()?.sync_all()?;
    Ok(())
}
