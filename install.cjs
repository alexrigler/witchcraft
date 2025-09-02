#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-var-requires */
const fs = require("fs");
const path = require("path");
const { exec } = require('child_process');

const rootPath = path.join(__dirname, "..");

// Note, console logs in this file only show up with the yarn `--verbose` option
console.log(`warp installer starting from rootPath ${rootPath}`);

const platform = process.env.npm_config_platform || process.platform;
const arch = process.env.npm_config_arch || process.arch;
console.log(`warp installer arch: ${arch}`);

const filename = `${name}-${version}-${platform}-${arch}`;
const fullPath = path.join(rootPath, "per-platform", filename);
console.error("build warp into", fullPath);
