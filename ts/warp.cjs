const warpNode = require('../target/release/warp.node');

module.warp = new warpNode.Warp("mydb.sqlite");
console.log("warp", module.warp);

export function search(query, threshold) {
    return module.warp.search(query, threshold);
}
