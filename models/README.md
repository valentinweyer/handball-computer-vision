# Local model cache

Model weights are local artifacts and are ignored by Git. The tracked logical
registry is `configs/models.yaml`; it records expected locations and known
checksums.

Several byte-identical checkpoint copies currently remain elsewhere in the
workspace. They are listed in the registry and have not been deleted.

## PRTReID source dependency

`prtreid/prtreid-soccernet-baseline.pth.tar` stores weights only. Loading it
also needs the BPBReID model definition, which `handball_cv.embeddings.prtreid`
imports from a source checkout (`prtreid/models/bpbreid.py`, `hrnet.py`,
`utils/constants.py`).

That checkout previously lived in `/tmp/handball-prtreid` and was lost to a
reboot, leaving a 378 MB checkpoint that could not be loaded and no record of
where it came from. Fetch it into the repo instead:

```bash
git clone --depth 1 https://github.com/VlSomers/prtreid.git prtreid-upstream
```

`prtreid-upstream/` is gitignored, like the other vendored upstream trees. Pass
it as `source_root`; scripts still default to the old `/tmp` path, so override
with `--prtreid-root prtreid-upstream`.
