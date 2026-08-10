# DataModule Tests

Historically, TorchGeo had unit tests for all data modules in this directory. However, these tests didn't actually ensure that the data modules were compatible with our tasks. Since then, almost all data module tests have been replaced by testing the data module directly with the task. This directory remains for historical purposes to test data modules that are not yet compatible with existing tasks. No new tests should be added to this directory. Instead, please add tests directly with the respective task.
