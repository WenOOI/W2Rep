# Open-source checklist

The release directory deliberately avoids deleting or modifying the research
workspace. Complete these items before publishing:

- [ ] Select and add the project license after confirming institutional policy.
- [ ] Add author names, affiliations, paper URL, and citation BibTeX.
- [ ] Publish checkpoints and record their SHA-256 hashes in the README.
- [ ] Replace example dataset and output paths in the YAML files.
- [ ] Run `pytest -q` in the intended public environment.
- [ ] Run a short multi-GPU training smoke test from a newly generated manifest.
- [ ] Reproduce one ImageNet, one video, and one ADE20K result from a published
      checkpoint.
- [ ] Verify that no dataset, credential, internal hostname, queue name, or
      private checkpoint is included in the Git history.
- [ ] Confirm redistribution terms for any example images or visualization
      assets added later.

