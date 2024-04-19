#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Find the optimal set of hyperparameters given experiment checkpoints."""

import glob
import json
import os
from collections import defaultdict

from tbparse import SummaryReader

OUTPUT_DIR = ""


# mypy does not yet support recursive type hints
def nested_dict() -> defaultdict[str, defaultdict]:  # type: ignore[type-arg]
    """Recursive defaultdict.

    Returns:
        a nested dictionary
    """
    return defaultdict(nested_dict)


if __name__ == "__main__":
    metrics = nested_dict()

    logs = os.path.join(OUTPUT_DIR, "logs", "*", "version_*", "events*")
    for log in glob.iglob(logs):
        hyperparams = log.split(os.sep)[-3]
        reader = SummaryReader(log)
        df = reader.scalars

        # Some event logs are for train/val, others are for test
        for split in ["train", "val", "test"]:
            rmse = df.loc[df["tag"] == f"{split}_RMSE"]
            mae = df.loc[df["tag"] == f"{split}_MAE"]
            if len(rmse):
                metrics[hyperparams][split]["RMSE"] = rmse.iloc[-1]["value"]
            if len(mae):
                metrics[hyperparams][split]["MAE"] = mae.iloc[-1]["value"]

    print(json.dumps(metrics, sort_keys=True, indent=4))
