# Python Template

A Python project template for VS Code.

## Requirements

- VS Code (with `Remote-Containers` extension by Microsoft)
- Docker (e.g. [Docker Desktop](https://www.docker.com/products/docker-desktop/))
- [GitHub CLI](https://cli.github.com/) (optional, recommended)
- [GNU Privacy Guard; GPG](https://gnupg.org/) (optional, recommended)

## About Environments

### Easy setup for devcontainer

Mac, Linux

```sh
.devcontainer/preCreateCommand.sh
```

Windows

```ps1
PowerShell -ExecutionPolicy RemoteSigned .\.devcontainer\preCreateCommand.ps1
```

If you do not use Github CLI or GnuPG, please modify `.devcontainer/docker-compose.yml` (and `.devcontainer/Dockerfile`, `.devcontainer/postCreateCommand.sh`).
Then **Reopen Folder in Container** with VS Code extension.

### Calculate TF-IDF

```sh
python -m ml_hands_on.tfidf
```

The results will be written in `output/`.

### Classify by XGBoost

```sh
python -m ml_hands_on.xgboost_classify
```

The potential accuracy will be shown in strout and the feature importances will be output to `output/`.

## Notes

- `dist/` is empty volume which masks original `dist/` if it exists.
- You cannot commit with GUI if GPG is activated. Use the integrated terminal instead.
- If you already created virtualenvs outside of container, please remove `.venv` before "Reopen Folder in Container".
- If you want to output debug log, set the environ variable `DEBUG` be truthy string ('0' is truthy here). On bash, `DEBUG=1 <command>`. On fish, `env DEBUG=1 <command>`.

## Known issues

- If you set `virtualenvs.create` to `false` in `postCreateCommand.sh`, then `poetry install` always updates most packages like this issue [#2079](https://github.com/python-poetry/poetry/issues/2079). As for now, it is set to `true` and this is not so bad.

## Production Container

See `.prod/`.
