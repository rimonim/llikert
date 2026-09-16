"""Operator command line: ``llikert serve``, ``llikert selfcheck``, ``llikert export-schemas``.

Server modules are imported lazily so the base client install stays lightweight.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import pathlib
import sys

TOKEN_ENV = "LLIKERT_API_TOKEN"


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=pathlib.Path, required=True, help="path to a GGUF file")
    p.add_argument("--model-sha256", help="expected SHA-256 of the GGUF file; startup fails on mismatch")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=512, help="physical micro-batch size (affects numerics)")
    p.add_argument("--flash-attn", choices=["auto", "on", "off"], default="off")
    p.add_argument("--kv-type", choices=["f16", "f32"], default="f32")
    p.add_argument("--threads", type=int, default=None)


def _require_server_extra() -> None:
    try:
        import fastapi  # noqa: F401
        import jinja2  # noqa: F401
        import numpy  # noqa: F401
        import pydantic  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        sys.exit(f"llikert: the server extra is not installed ({exc.name}); install 'llikert[server]'")


def _load_service(args: argparse.Namespace, limits=None):
    from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig
    from llikert.server.service import Limits, ScoringService

    adapter = LlamaAdapter(
        LlamaConfig(
            model_path=args.model,
            device=args.device,
            n_ctx=args.n_ctx,
            batch_size=args.batch_size,
            flash_attn=args.flash_attn,
            kv_type=args.kv_type,
            threads=args.threads,
            expected_sha256=args.model_sha256,
        )
    )
    try:
        service = ScoringService(adapter, limits or Limits())
        service.smoke_check()
    except Exception:
        adapter.close()
        raise
    return service


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def cmd_serve(args: argparse.Namespace) -> int:
    _require_server_extra()
    import uvicorn

    from llikert.server.app import AuthMode, HttpSettings, create_app
    from llikert.server.service import Limits

    auth = AuthMode(args.auth)
    token = None
    if auth == AuthMode.NONE and not _is_loopback(args.host):
        sys.exit("llikert: --auth none is only allowed on a loopback address; use --auth token or --auth platform")
    if auth == AuthMode.TOKEN:
        token = os.environ.get(TOKEN_ENV)
        if not token or len(token) < 16:
            sys.exit(f"llikert: --auth token requires {TOKEN_ENV} with at least 16 characters")
    if auth == AuthMode.PLATFORM:
        logging.getLogger("llikert.server").warning(
            "auth=platform: this service performs no authentication; the hosting platform must enforce it"
        )

    limits = Limits(max_items_per_request=args.max_items, max_body_bytes=args.max_body_bytes, queue_limit=args.queue)
    settings = HttpSettings(auth=auth, token=token, max_body_bytes=args.max_body_bytes, queue_limit=args.queue)
    app = create_app(lambda: _load_service(args, limits), settings)
    # one process, one worker: the model is loaded exactly once
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level, access_log=False)
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    _require_server_extra()
    report: dict = {"checks": []}

    def record(name: str, ok: bool, detail: str | None = None) -> None:
        report["checks"].append({"check": name, "ok": ok, **({"detail": detail} if detail else {})})

    try:
        from llikert.server import native

        lib = native.Native()
        record("native_library", True, f"llama.cpp {lib.llama_cpp_version}")
    except Exception as exc:  # noqa: BLE001
        record("native_library", False, str(exc))
        print(json.dumps(report, indent=2))
        return 1
    try:
        service = _load_service(args)
        record("model_and_template", True, service.profile["name"])
        record("smoke_check", True)
        report["engine_fingerprint"] = service.fingerprint
        report["engine"] = service.identity
        service.adapter.close()
    except Exception as exc:  # noqa: BLE001
        record("model_and_template", False, f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2))
    return 0 if all(c["ok"] for c in report["checks"]) else 1


def cmd_export_schemas(args: argparse.Namespace) -> int:
    _require_server_extra()
    from llikert.server.schemas import export_schemas

    return export_schemas(args.out, check=args.check)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llikert")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the scoring service")
    _add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--auth", choices=["none", "token", "platform"], default="none")
    serve.add_argument("--max-items", type=int, default=64)
    serve.add_argument("--max-body-bytes", type=int, default=8 * 1024 * 1024)
    serve.add_argument("--queue", type=int, default=4)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    check = sub.add_parser("selfcheck", help="diagnose native libraries, model, template, and device")
    _add_engine_args(check)
    check.set_defaults(func=cmd_selfcheck)

    schemas = sub.add_parser("export-schemas", help="write JSON schemas and OpenAPI to a directory")
    schemas.add_argument("--out", type=pathlib.Path, default=pathlib.Path("schemas"))
    schemas.add_argument("--check", action="store_true", help="fail if committed schemas differ")
    schemas.set_defaults(func=cmd_export_schemas)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
