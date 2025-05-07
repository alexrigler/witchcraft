use std::path::PathBuf;
use std::fs::File;
use csv::ReaderBuilder;
use serde::Deserialize;
use rand::prelude::*;
use rusqlite::*;

//use candle_transformers::models::t5;
mod t5;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D, IndexOp};
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

#[derive(Debug, Deserialize)]
struct Record {
    _line: u32,
    text: String,
}

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
}

fn kmeans(data: &Tensor, k: usize, max_iter: u32, device: &Device) -> Result<(Tensor, Tensor)> {
    println!("kmeans {}", data);
    let (n, _) = data.dims2()?;
    println!("kmeans {}", n);
    let mut rng = rand::rng();
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    let centroid_idx = indices[..k]
        .iter()
        .copied()
        .map(|x| x as u32)
        .collect::<Vec<_>>();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((n,), DType::U32, device)?;
    for _ in 0..max_iter {
        let dist = cdist(data, &centers)?;
        cluster_assignments = dist.argmin(D::Minus1)?;
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
            let cluster_data = data.index_select(&indices, 0)?;
            let mean = cluster_data.mean(0)?;
            centers_vec.push(mean);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    Ok((centers, cluster_assignments))
}


fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let dims = tensor.dims();
    let num_rows = dims[0];

    // Collect each row as a separate Tensor of shape [128]
    (0..num_rows)

        .map(|i| {
            let row_tensor = tensor.i(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })

        .collect()
}

fn stack_tensors(vectors: Vec<Tensor>) -> Tensor {
    Tensor::cat(&vectors, 0).unwrap() // `0` means stacking along rows (axis 0)
}

fn gather_embeddings(model: t5::T5EncoderModel, tokenizer: Tokenizer) -> Result<Tensor> {

    let file = File::open("collection.tsv")?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let mut doc_embedding = vec![];
    for result in rdr.deserialize() {
        let record: Record = result?;
        let now = std::time::Instant::now();
        let tokens = tokenizer
            .encode(record.text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
        let embeddings = model.forward(&token_ids)?;
        let elapsed_time = now.elapsed();
        println!("Running slow_function() took {} ms.", elapsed_time.as_millis());
        let normalized = normalize_l2(&embeddings)?.get(0)?;
        let split = split_tensor(&normalized);
        doc_embedding.extend(split);
    }
    Ok(stack_tensors(doc_embedding))
}

struct DB {
    connection: rusqlite::Connection,
}

impl DB {
    pub fn new() -> Self {
        let connection = Connection::open_in_memory().unwrap();
        Self { connection }
    }

    pub fn init(self: &Self) -> Result<()> {
        //unsafe {
            //let _guard = LoadExtensionGuard::new(&self.connection)?;
            //self.connection.load_extension("trusted/sqlite/extension", None)?
        //}

        println!("init");
        let query = "CREATE TABLE document(filename TEXT, hash TEXT);";
        self.connection.execute(query, ()).unwrap();
        Ok(())
    }

    fn add_doc(self: &Self, filename: &str, hash: &str) -> Result<()> {

        self.connection.execute("INSERT INTO document VALUES(?1, ?2)", (&filename, &hash))?;

        Ok(())
    }

}

fn main() -> Result<()> {

    let db = DB::new();
    db.init().unwrap();
    db.add_doc("foobar.txt", "").unwrap();
    let (builder, tokenizer) = T5ModelBuilder::load()?;
    let model = builder.build_encoder()?;

    let matrix = gather_embeddings(model, tokenizer)?;
    let device = Device::Cpu;
    let (_, idxs) = kmeans(&matrix, 16, 5, &device)?;
    println!("idxs {}", idxs);

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
