# Uploading this prepared repository

This folder contains the code/evidence submission, the consolidated notebook and supporting slides. The dissertation PDF and recorded presentation video are submitted separately through the university's required route.

## Local checks

From the repository root:

```sh
python tools/verify_saved_results.py
python tools/verify_notebook.py
git status --short
```

There are no model weights or raw image collections to upload. `.gitignore` excludes environments, credentials, new training runs and common large model assets. Evidence files must retain their recorded bytes; `.gitattributes` prevents newline conversion in the fingerprinted evidence. Do not export CSVs through a spreadsheet application before uploading them.

## Publish with Git

Create an empty repository in your own GitHub account with the required visibility. Do not initialise a second README or licence remotely when uploading this existing folder. Then run the following, replacing the remote URL with the one GitHub gives you:

```sh
git add .
git commit -m "Prepare MSc project code, notebook and evidence"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

The folder already has a local Git repository on `main`. If `origin` has subsequently been configured, check `git remote -v` and use that existing remote rather than adding it twice. Configure your own Git author identity if Git requests it.

Alternatively, use GitHub Desktop to add this existing local repository and publish it. Include dotfiles such as `.github`, `.gitignore` and `.gitattributes`; they supply checks and preservation rules.

## After upload

- Confirm the README, notebook and presentation links work.
- Confirm the **Verify submission** workflow passes.
- Ensure markers can access the repository; private repositories need the required collaborators/access route.
- Put the real repository URL into the final report/submission details.

The author has not selected an open-source software licence. Third-party dataset, imagery and model terms remain applicable. No institutional ethics outcome or external-asset sharing permission is inferred by packaging this repository.
