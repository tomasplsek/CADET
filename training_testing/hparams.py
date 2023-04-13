import os

for lr in [0.0005]: #0.002, 0.001, 0.0005]:
    for batch in [24]:
        for network in [""]: #, "_relu_Stan"]: # "_drop_Stan_normal_new"]: #, "_activated_Stan", "_maxpool_Stan", "_activated_maxpool_Stan"]: #, "_50_alphaweights_Stan"]:
            for cavities in ["_100"]: #"_50", "_90","_100"]:
                for data in ["_normal_final"]:
                    for drop in [0.2]: #0.0, 0.2]:
                        # if os.path.exists(f"logs/b{batch}_lr{lr}_d{drop}{cavities}"): continue
                        os.system(f"python3 train_CADET.py {lr} {batch} {network+cavities} {data} {drop}")