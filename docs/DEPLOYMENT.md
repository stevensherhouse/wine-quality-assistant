# Container and cloud handoff

The repository contains a Dockerfile, health check, and CI workflow. No cloud service has been provisioned and no hosted URL is claimed. Local Docker verification requires a running Docker engine.

## Container

The default, no-API-fee path uses the local Qwen model. From the repository root, download the model with `python -m src.setup_local` first, then:

```sh
docker compose up --build
```

Open `http://localhost:8501`. The Compose recipe runs a CPU-only llama.cpp model server and the app in a shared network namespace. The LLM listens on loopback port 8080, and only the app's port 8501 is published, bound to localhost. The downloaded GGUF is mounted read-only. This portable recipe may be much slower than the native Windows GPU setup; it has not been container-tested here. Do not run it while the native app already occupies port 8501.

The app image installs dependencies and trains during build; allow internet access to UCI and Python package indexes, plus sufficient memory for the tree models. It contains the trained regression model but no credentials or LLM weights. The build context excludes all raw data, binaries, local tracking storage, and `.env`. Model weights enter through the Compose mount. A clean build recreates regression experiments. The llama.cpp `server` image is documented by its upstream project; pin an image digest before a production deployment.

For optional Nebius mode, the app alone can run with `docker build -t wine-quality-assistant .` followed by `docker run --rm -p 127.0.0.1:8501:8501 --env-file .env -e LLM_PROVIDER=nebius wine-quality-assistant`. Supply credentials privately and expect provider usage charges. Do not put secrets in build arguments.

## Cloud container service

Cloud hosting has separate infrastructure costs even if the model has no API fee. For a local-model deployment, the host must also run llama.cpp with the GGUF weights and enough memory; the app expects its model API on loopback in the same network namespace. The Compose recipe illustrates that arrangement. A basic web-only container hosting plan is insufficient for the whole local-model stack.

Alternatively, build and push the app image and select optional `LLM_PROVIDER=nebius`. Configure port **8501**, HTTP health path **/_stcore/health**, and provider-managed secrets for `NEBIUS_API_KEY`, `NEBIUS_MODEL`, and optionally `NEBIUS_BASE_URL`. Configure the platform's access controls before sharing an endpoint that uses your paid provider key. Set reasonable CPU and memory limits; actual sizing requires a load test. No cloud deployment has been performed.

The app reads the baked model and does not retrain at startup. This makes restarts independent of the UCI download. A new model release requires a new build. CI retrains and runs tests for pushes and pull requests, but does not automatically publish or deploy an image.

## Remote experiment tracking

Set `MLFLOW_TRACKING_URI` to an authenticated MLflow service before running training. Configure the MLflow server's object-store artifact destination on that server and supply cloud credentials through the platform's secret store. Each run records dataset content hashes, configuration, all estimator/preprocessing parameters, validation/test metrics, and a model artifact.

The default SQLite database and `mlruns/` are local only and excluded from Git. They are suitable for a single-machine capstone demonstration; they do not provide shared or durable cloud tracking. To preserve local history, back up the database and artifact directory together. Absolute artifact locations may need migration when moving to another machine.

## Production improvements

Add request rate limits, authentication, provider budget controls, persistent monitoring, explicit model-release approvals, and broader evaluation of LLM extraction. Evaluate drift on new laboratory samples and performance by wine type before expanding scope. Marginal per-feature bounds do not detect every unrealistic combination of measurements.
