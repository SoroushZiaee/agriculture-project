import argparse
from typing import Dict


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_target_path",
        type=str,
        default="/home/szsoroush/Human_Car_detection/notebooks/iran-test2-vehicle-counting-night.mp4",
        help="target path",
    )

    opts = parser.parse_args()
    return opts
    
def run():
    pass

def main(conf: Dict):
    run()

if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    main(conf)

    
