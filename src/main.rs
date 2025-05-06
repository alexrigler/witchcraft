use std::path::PathBuf;

//use candle_transformers::models::t5;
mod t5;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
//use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load() -> Result<(Self, Tokenizer)> {
        let device = Device::Cpu;
        let path = PathBuf::from(r"/Users/jhansen/src/xtr-warp/foo.safetensors");
        let weights_filename = vec![path];
        let config = std::fs::read_to_string("/Users/jhansen/src/xtr-warp/xtr-base-en/config.json")?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file("/Users/jhansen/src/xtr-warp/xtr-base-en/tokenizer.json").map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

}

fn main() -> Result<()> {
    let (builder, tokenizer) = T5ModelBuilder::load()?;
    let model = builder.build_encoder()?;

    let now = std::time::Instant::now();
    let tokens = tokenizer
        .encode("do buildings change size due to weather?", true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
    println!("ids {}", token_ids);
    let embeddings = model.forward(&token_ids)?;
    let elapsed_time = now.elapsed();
    println!("Running slow_function() took {} ms.", elapsed_time.as_millis());
    println!("\nembeddings:\n{}", embeddings);
    println!("\nembeddings l2:\n{}", normalize_l2(&embeddings).unwrap());

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
