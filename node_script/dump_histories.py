#!/usr/bin/env python3


import argparse

from plug_insert.common import History, project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="number of sampling point")
    args = parser.parse_args()

    rosbag_base_path = project_path() / "rosbag"

    for path in rosbag_base_path.iterdir():
        if path.is_symlink():
            continue
        if not path.name.endswith(".bag"):
            continue
        history = History.from_bag(path, args.n)
        history.dump()

    histories = History.load_all()
