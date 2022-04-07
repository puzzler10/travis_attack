import argparse
from fastcore.basics import in_jupyter
print(in_jupyter())
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--acc_steps", type=int)
parser.add_argument("--reward_fn")
args = parser.parse_args()
print(args)
print(vars(args))

def args2dict(args):
    d = dict()
    for o in args.args: 
        k, v = o.split("=")
        d[k]=v
    return d 

print(args2dict(args))
newargs = args2dict(args)
cfg = dict(lr=5,acc_steps=2,pp_length=20)
print(cfg, newargs)
for k,v in newargs.items(): 
  cfg[k]=v
print(cfg)

