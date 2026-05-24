# LiteLLM Wrapper — DigitalOcean Example

A small wrapper and configuration for running LiteLLM-based tools on a DigitalOcean environment.

This repository includes a minimal example of configuration and a simple wrapper script to run or integrate LiteLLM models locally or on a droplet/container.

**Contents**
- `config.example.env` : example environment variables for runtime configuration.
- `litellm_config.yaml` : LiteLLM configuration file (model, device, and runtime options).
- `litellm_wrapper.py` : thin Python wrapper to load configuration and run the LiteLLM model.
- `models_cache.json` : optional cache describing downloaded or available models.

## Overview

This project demonstrates how to configure and run a LiteLLM-based application in a lightweight, reproducible way. It is intended as a starting point for deploying or experimenting with local LLM inference on DigitalOcean droplets, containers, or other Linux hosts.

Key goals:
- Provide a simple, documented config and example env file.
- Offer a minimal wrapper to load config and start inference.
- Keep dependencies and setup minimal so it is easy to deploy.

## Requirements

- Python 3.10+ recommended
- LiteLLM and any model runtime backends required by your chosen model (see `litellm_config.yaml`)

Install typical Python dependencies with pip (adjust packages to match your environment):

```bash
python -m pip install -r requirements.txt
```

If there is no `requirements.txt`, install the LiteLLM packages you need, e.g.:

```bash
python -m pip install litellm
```

## Configuration

1. Copy `config.example.env` to `.env` and update values as needed.
2. Review `litellm_config.yaml` to select model, device (cpu/gpu), and other runtime options.

Common env variables to set (example keys found in `config.example.env`):
- `MODEL_PATH` or model identifier
- `DEVICE` (cpu, cuda)
- `PORT` (if running a web service)

## Usage

Run the wrapper to start the model using the provided configuration:

```bash
# ensure env is loaded (example using direnv or dotenv)
python litellm_wrapper.py
```

The wrapper reads `litellm_config.yaml` and the environment variables to initialize the model. Consult the top of `litellm_wrapper.py` for runtime flags and options.

## Models cache

`models_cache.json` can be used to track downloaded model files or metadata. It is optional and not required for basic runs.

## Deployment notes (DigitalOcean)

- Use a small droplet with CPU-only models, or a droplet with GPU if your model requires CUDA acceleration.
- Containerize the app (Docker) for reproducible deployments.
- Persist model files on disk or attach block storage to avoid repeated downloads.

## Contributing

PRs welcome. If you add a `requirements.txt` or Dockerfile, please update this README with run and build instructions.

## License

This project does not include a license file; add one if you intend to publish or share the code publicly.
