# DataModule Tests

Historically, TorchGeo had unit tests for all data modules in this directory. However, these tests didn't actually ensure that the data modules were compatible with our trainers. Since then, almost all data module tests have been replaced by testing the data module directly with the trainer. This directory remains for historical purposes to test data modules that are not yet compatible with existing trainers. No new tests should be added to this directory. Instead, please add tests directly with the respective trainer.
