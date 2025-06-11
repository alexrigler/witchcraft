cargo run --release --features accelerate querycsv \
    $HOME/src/xtr-warp/beir/nfcorpus/questions.test.tsv output.txt &&\

python score.py nfcorpus/output.txt $HOME/src/xtr-warp/beir/nfcorpus/collection_map.json /Users/jhansen/src/xtr-warp/beir/nfcorpus/qrels.test.json
