"""Return exactly the acquired eight-drop readouts to three donor bodies."""
import gzip
import json
import numpy as np
from .tt24_donor_trial import Native, World, ROOT, DESIGNS


def main():
    weights,names=[],[]
    for index in range(4):
        with gzip.open(ROOT/f"donor-adapt-{index}.json.gz","rt") as src:
            trained=json.load(src)
        for mode in ("learned","label_shuffle","isotropic","frozen"):
            names.append((index,mode))
            weights.append(trained["outcomes"][mode]["weights"])
    rng=np.random.default_rng(1471301991)
    native=Native()
    donors=[]
    try:
        for index,design in enumerate(DESIGNS):
            world=World(native,design)
            try:
                seeds=rng.integers(1,2**31-1,24).tolist()
                batch=world.prepare(seeds,"held")
                outcome=world.play(weights,batch,cap=300)
                donors.append(dict(index=index,seeds=seeds,metrics=outcome.tolist()))
            finally:
                world.close()
    finally:
        native.working.cleanup()
    with gzip.open(ROOT/"donor-adapt-retention.json.gz","wt") as out:
        json.dump(dict(names=names,donors=donors,source_practice_after_target=0,
                       metrics=["total","longest","ticks","contacts"]),out,separators=(",",":"))
    matrix=np.asarray([d["metrics"] for d in donors])[:,:,:,1].mean(axis=(0,2)).reshape(4,4)
    print(matrix,flush=True)


if __name__=="__main__":
    main()
