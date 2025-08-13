
xtr-base-en/model.safetensors:
	python downloadxtr.py

assets/config.json.zst assets/tokenizer.json.zst assets/xtr.safetensors.zst: xtr-base-en/model.safetensors
	python downloadweights.py

download: assets/config.json.zst assets/tokenizer.json.zst assets/xtr.safetensors.zst

build: download
	cargo build --release --features accelerate
	ln -vf target/release/libwarp.dylib target/release/warp.node

buildemb: download
	cargo build --release --features accelerate,embed-assets
	ln -vf target/release/libwarp.dylib target/release/warp.node

run: build
	node index.js

mcp:
	yarn build
	cmcp "yarn start" tools/call name=search 'arguments:={"q": "is there a connection between milk intake and pimples in young people?" }'
