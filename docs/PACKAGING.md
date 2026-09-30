# Packaging notes and known limitations

Prepared 22 September 2026 from the existing project folders. The original repository HEAD was `11fa333eb8dc332bf0e0f4425cc8d4187ae499ba`. This submission copy also includes the previously untracked RoBERTa plotting script/figures and validation prediction exports. Original working directories and Git history were not changed.

Packaging changes: added a repository overview, evidence map and reproduction guide; corrected the outdated v2 completion status; cleared the original methodology notebook execution outputs; made the source-disjoint runner roots relocatable and remapped archived image paths; added read-only result verification and ignore rules. No model was retrained, result adjusted, threshold retuned or metric replaced.

The standalone historical report-generation helper was omitted from the runnable pipeline. Its full source is now preserved in the consolidated notebook historical appendix, alongside the earlier local source-disjoint runner and methodology setup. Draft reports, credentials, environments and large assets were excluded. The `.git` history was not copied: this is a clean submission snapshot, with provenance recorded above.

Known limitations retained transparently:

- `scripts/build_multimodal_case_studies.py` has an indentation defect in the SHAP token summarisation loop. The saved selected-case CSV has blank token-summary columns. Raw token exports remain available; the report appendix uses those. This historical script/result pair was not silently changed during packaging.
- Forty-five perceptual duplicate candidates remain unreviewed; do not claim event-level or complete visual independence.
- Source-disjoint validation/test use only two/three source communities. Bootstrap intervals condition on the recorded checkpoints and cases, not retraining or new-community variation.
- Archived requirements may need platform-specific resolution. No claim is made that installing the frozen environment on arbitrary hardware succeeds.
- Full experiment execution requires omitted data and model assets. The consolidated notebook supports Run All inspection using embedded evidence; the original methodology notebook still requires external assets. Standard-library verification commands are available without Jupyter.

Before attaching a GitHub link to the report, create the remote repository, upload this folder and verify that markers can access it. A private repository requires explicit access for markers. Publishing has not been performed by the packaging step. Review third-party data/imagery terms and your own AI/contribution disclosure before choosing public visibility; no licence or institutional approval has been inferred.

## Notebook integration — 29 September 2026

The complete project notebook is included at the repository root and linked from the README. Its 42 Python source files and nine registered methodology code blocks (eight current cells and the earlier setup variant) were checked against the implementations. A fresh-kernel inspection run compiled the registered code and verified the 198 evidence fingerprints and both sets of saved predictions; no training or model inference was performed.

The notebook embeds a frozen evidence/documentation snapshot from its assembly. Current root documentation is authoritative for repository setup and upload instructions. This separation avoids recursively embedding the notebook inside itself. `tools/verify_notebook.py` checks the embedded scientific evidence and canonical experiment source against the current repository. Later changes to the experiment code or evidence must update the notebook as well; documentation-only changes do not require replacing its archived provenance notes.

No remote URL, software licence or ethics-review outcome has been invented. The prepared folder can be committed and uploaded; remote publication and marker access remain separate actions.
