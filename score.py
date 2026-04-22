import json
import math
import sys
import pytrec_eval


results_fn = sys.argv[1]
map_fn = sys.argv[2]
qrel_fn = sys.argv[3]

results = open(results_fn, "r")

with open(map_fn, 'r') as file:
    remap = json.load(file)

with open(qrel_fn, 'r') as file:
    qrel = json.load(file)

run = {}

for line in results:
    k, vs = line[:-1].split("\t")
    run[k] = {}

    result = []
    for (i, v) in enumerate(vs.split(",")):
        if v:
            if remap is not None:
                name = remap[v]
            else:
                name = v
            run[k][name] = float(-i)
            result.append(name)

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut'})


eval = evaluator.evaluate(run)

total = 0
for k in eval.keys():
    total += (eval[k]["ndcg_cut_10"])

print (total / len (eval.keys()))
