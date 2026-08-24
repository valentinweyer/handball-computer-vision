# Local model cache

Model weights are local artifacts and are ignored by Git. The tracked logical
registry is `configs/models.yaml`; it records expected locations and known
checksums.

Several byte-identical checkpoint copies currently remain elsewhere in the
workspace. They are listed in the registry and have not been deleted.
