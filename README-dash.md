This is a from-scratch Rust reimplementation of Stanford's XTR-Warp
semantic search engine (see https://github.com/jlscheerer/xtr-warp and
https://dl.acm.org/doi/10.1145/3726302.3729904 for more details.)

The main development has so far happened in
https://github.com/jhansen_dbx/rust-warp , where you can find the full project
history and additional command line tools for, e.g., load testing, and
for downloading and converting the XTR neural network weights from Huggingface.

Unfortunately, this has to be a private repo, so you will have to ping jhansen
to get access.

To build, you will need to install the Rust toolchain, see
https://www.rust-lang.org/tools/install. You will need both the ARM and x86 targets
installed:

```sh
rustup target add arch64-apple-darwin
rustup target add x86_64-apple-darwin
```

targets, to be able to build a universal binary. You will also need Xcode
installed.

It is possible to cross-compile from macOS to Windows, if for some reason you
want to do that, you will need to install xwin and at least one Windows target
with:

```sh
cargo install xwin --locked
rustup target add x86_64-pc-windows-msvc
```
