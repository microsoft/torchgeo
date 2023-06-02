#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil


def delete_scene(directories: list[str], scene_id: str) -> None:
    """Delete scene_id from all directories.

    Args:
        directories: directories to check
        scene_id: scene to delete
    """
    print(f"Removing {scene_id}")
    for directory in directories:
        scene = os.path.join(directory, scene_id)
        if os.path.exists(scene):
            shutil.rmtree(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", help="directories to compare")
    parser.add_argument(
        "--delete-different-locations",
        action="store_true",
        help="delete scene locations that do not match",
    )
    parser.add_argument(
        "--delete-different-dates",
        action="store_true",
        help="delete scene dates that do not match (must be same satellite)",
    )
    args = parser.parse_args()

    print("Computing sets...")
    scene_sets = [set(os.listdir(directory)) for directory in args.directories]

    print("Computing union...")
    union = set.union(*scene_sets)
    total = len(union)

    print("Computing intersection...")
    intersection = set.intersection(*scene_sets)
    remaining = len(intersection)

    print("Computing difference...")
    difference = union - intersection
    delete_locations = len(difference)

    if args.delete_different_locations:
        for scene_id in difference:
            delete_scene(args.directories, scene_id)

    delete_times = 0
    for scene_id in intersection:
        time_sets = set.intersection(
            *[
                set(os.listdir(os.path.join(directory, scene_id)))
                for directory in args.directories
            ]
        )
        if len(time_sets) != 4:
            if args.delete_different_dates:
                delete_times += 1
                delete_scene(args.directories, scene_id)

    remaining -= delete_times
    delete = delete_locations + delete_times
    if not (args.delete_different_locations or args.delete_different_dates):
        print(f"Would delete {delete} scenes, leaving {remaining} remaining scenes.")
    else:
        print(f"Deleted {delete} scenes, leaving {remaining} remaining scenes.")
