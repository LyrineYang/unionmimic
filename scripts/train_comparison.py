import argparse
from pathlib import Path
from unionmimic.comparison import run

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true')
    run(p.parse_args())
