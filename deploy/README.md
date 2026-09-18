# Setting up the LLikert scoring service

This guide is for the person who sets up the scoring service for a research group, usually lab IT support or a technically comfortable lab member. You should be comfortable running commands in a Linux terminal; no programming is needed.

The researchers themselves only install a small R or Python package. They need two things from you: the **service address** and an **access key**.

Setup takes about an hour, mostly waiting for downloads and builds.

## What you need

- **A Linux machine with an NVIDIA graphics card with at least 24 GB of memory.** The verified card is an RTX 4090. Other 24 GB cards (L4, A10G, RTX A5000 and similar) should work but must pass the check in step 6 before they are used for research.
- **An NVIDIA driver, version 575.51 or newer.** `nvidia-smi` shows the driver version.
- **Docker** and the **NVIDIA Container Toolkit**.
- **About 45 GB of free disk space:** roughly 25 GB for preparing the model file, which is 16 GB when done, plus 5 GB for the service image.

Check that Docker can use the graphics card:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

If this prints a table with your graphics card, you are ready. If it fails right after you installed the NVIDIA Container Toolkit, restart Docker (`sudo systemctl restart docker`) and try again.

## 1. Get the code

```bash
git clone https://github.com/rimonim/llikert.git
cd llikert
```

All later commands are run from this `llikert` folder.

## 2. Prepare the model file

The service uses the language model Qwen3-4B-Instruct-2507 in a specific file format. The script downloads the official model (about 8 GB), converts it, and checks that the result matches the verified file exactly:

```bash
deploy/prepare-model.sh models
```

This takes 10–40 minutes, depending on your connection. It should end with:

```
OK: .../models/qwen3-4b-instruct-2507-f32.gguf (sha256 a5733d5a25e8…)
```

If it reports `MISMATCH`, don't use the file; see [troubleshooting](../docs/troubleshooting.md).

## 3. Build the service image

```bash
docker build -f deploy/Dockerfile.cuda -t llikert-service:0.1.0.dev0-cuda12.9 \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
```

This takes about 15 minutes the first time. Only the code and its fixed dependencies go into the image; the model file stays outside.

## 4. Create an access key

```bash
python3 -c "import secrets; print(secrets.token_hex(32))" > llikert-access-key.txt
chmod 600 llikert-access-key.txt
```

Anyone with this key can use the service. Share it only with the researchers who should have access. To revoke access, create a new key and run `deploy/service.sh recreate`.

## 5. Start the service

```bash
deploy/service.sh start
```

The first time, this creates the container; afterwards it just starts it again. It waits until the service is ready, which takes about 20 seconds, and prints the address:

```
created container 'llikert' from llikert-service:0.1.0.dev0-cuda12.9
starting 'llikert' (loading the model takes about 20 seconds)
ready after 21s; address: http://127.0.0.1:8080
```

`deploy/service.sh` is the only command you need for running the service; the sections below use it throughout. If the service never becomes ready, run `deploy/service.sh logs`. When the model file is wrong or the graphics card is unusable, the service stops with an explanation instead of hanging.

**Letting researchers reach it.** By default the service accepts connections only from this machine (`127.0.0.1`). Choose one option:
- **Researchers work on this machine:** nothing more to do. The address is `http://127.0.0.1:8080`.
- **Researchers connect from their own computers:** put the service behind your institution's HTTPS reverse proxy, the same way you would publish any internal web application, and give researchers the `https://` address. The service does not encrypt traffic itself, so don't publish port 8080 directly on a network.
- **A quick test from another computer:** use an SSH tunnel (`ssh -L 8080:127.0.0.1:8080 user@this-machine`); the address is then `http://127.0.0.1:8080` on that computer.

## 6. Check the results on this machine

Every graphics card, driver and library version counts as a different setup. Run the accuracy check once on your machine before researchers use it for studies. It scores 450 prepared texts and compares them with reference results computed from the original model:

```bash
python3 -m venv check-env
check-env/bin/pip install ./python
LLIKERT_TOKEN="$(cat llikert-access-key.txt)" check-env/bin/python tools/fidelity_gate_http.py \
  --url http://127.0.0.1:8080 --json fidelity-check.json
```

The check takes about a minute. The output starts with `"passed": true` if the setup is fine. Keep `fidelity-check.json`: it records the exact setup (the engine fingerprint) that was checked. If it prints `"passed": false`, do not use the service for research, and report the file to the LLikert maintainers.

## Stopping the service (and freeing the graphics card)

**While the container runs, it holds about 17 GB of graphics memory, even when nobody is scoring.** Stop it whenever you need the card for something else.

```bash
deploy/service.sh stop      # stop the service and free the graphics card
deploy/service.sh start     # start it again and wait until it is ready
deploy/service.sh status    # running? ready? how much graphics memory? recent activity?
```

Things worth knowing:
- **Stopping is safe.** The service stores nothing. Researchers who are scoring will see a "did not become ready" or connection error; work saved in their checkpoints resumes when you start the service again.
- **`status` shows recent activity,** so you can check whether anyone is scoring before you stop it. `stop` also warns you if there were requests in the last five minutes.
- **The service stays stopped** until you run `deploy/service.sh start` again, including across reboots. While it is running, it restarts by itself after a reboot or a crash.
- **Starting takes about 20 seconds** for the model file to be hashed, loaded and warmed up.

## 7. Give researchers access

Send each researcher:
1. **The service address**, from step 5.
2. **The access key**, from `llikert-access-key.txt`, through a secure channel, not plain email or chat if you can avoid it.
3. **A link to the guide for their language:** [R](../r/llikert/README.md) or [Python](../python/README.md).

## Day-to-day

| Task | Command |
|---|---|
| See whether it is running and ready | `deploy/service.sh status` |
| Stop it and free the graphics card | `deploy/service.sh stop` |
| Start it again | `deploy/service.sh start` |
| Restart it | `deploy/service.sh restart` |
| Look at the log | `deploy/service.sh logs 50` (the log never contains research texts or keys) |
| Apply a new key, image or setting | `deploy/service.sh recreate` |
| Change the access key | Create a new key (step 4), then `deploy/service.sh recreate` |
| Update to a new version | `git pull`, rebuild (step 3), `deploy/service.sh recreate`, then check again (step 6) |
| Check the model and graphics card without serving | `deploy/service.sh selfcheck` |

Things to know:
- **Give the service the graphics card to itself.** Another program taking graphics memory while it runs can make it crash, and the service holds its own memory until you stop it (see above). Researchers' saved progress is not lost; they can resume after a restart.
- **Settings are fixed on purpose.** The settings that affect results are built into the image. Changing the model file or the image changes the setup, and researchers will be told to re-prepare their tasks.
- **Several researchers can use the service at once.** Requests wait in a short queue, and researchers' packages retry automatically when it is busy.
- **Speed:** roughly 15 texts per second on an RTX 4090, one request at a time.

## Settings reference

The defaults suit the setup in this guide. To change one, copy `deploy/llikert.env.example` to `deploy/llikert.env`, edit it, and run `deploy/service.sh recreate`.

| Setting | Default | Meaning |
|---|---|---|
| `LLIKERT_MODELS_DIR` | `./models` | Folder holding the model file |
| `LLIKERT_MODEL_FILE` | `qwen3-4b-instruct-2507-f32.gguf` | The model file to load |
| `LLIKERT_MODEL_SHA256` | the verified checksum | The service refuses to start if the file does not match |
| `LLIKERT_KEY_FILE` | `./llikert-access-key.txt` | File holding the access key |
| `LLIKERT_PORT` | `8080` | Port on this machine |
| `LLIKERT_BIND` | `127.0.0.1` | Which addresses may connect; use `0.0.0.0` only behind an HTTPS reverse proxy |
| `LLIKERT_N_CTX` | `4096` | Longest prompt, in tokens (about 3,000 words). Raising it changes the setup and uses more graphics memory |
| `LLIKERT_IMAGE` | `llikert-service:0.1.0.dev0-cuda12.9` | Image to run |
| `LLIKERT_CONTAINER` | `llikert` | Container name, if you run more than one service |

Settings that would change the scores (graphics card use, batch size, attention and precision) are fixed in the image and cannot be changed here.

## Privacy and security

- **What the service keeps:** it holds research texts only while scoring them, and stores no texts, results or keys. Its logs record only counts, statuses and timings.
- **Container setup:** it runs as an unprivileged user and reads the model folder read-only.
- **Access:** anyone with the access key can send texts to the service and use your graphics card. Keep the key private.
- **Institutional rules:** if your institution restricts where research data may be processed, this machine counts as a place where it is processed.

## Other setups

- [`hf-endpoint.md`](hf-endpoint.md): running the service on Hugging Face's paid cloud service. Experimental and not yet tested.
- [`../docs/local-service.md`](../docs/local-service.md): installing without Docker.
- [`../docs/troubleshooting.md`](../docs/troubleshooting.md): problems and solutions.
