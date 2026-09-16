"""M0 container probe: load the model in a background thread; /health is 503 until ready."""

import ctypes
import http.server
import json
import os
import threading

import native

state = {"ready": False, "error": None, "info": {}}


def load():
    try:
        lib = native.load()
        lib.llama_backend_init()
        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = -1
        model = lib.llama_model_load_from_file(os.environ["LLIKERT_MODEL"].encode(), mp)
        if not model:
            raise RuntimeError("model load failed")
        buf = ctypes.create_string_buffer(256)
        lib.llama_model_desc(model, buf, len(buf))
        state["info"] = {"desc": buf.value.decode(), "gpu_offload": lib.llama_supports_gpu_offload()}
        state["ready"] = True
    except Exception as e:  # noqa: BLE001
        state["error"] = type(e).__name__


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            code = 200 if state["ready"] else 503
            body = b"{}"
        elif self.path == "/probe":  # probe only; the real service authenticates this kind of route
            code, body = 200, json.dumps(state).encode()
        else:
            code, body = 404, b"{}"
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


threading.Thread(target=load, daemon=True).start()
http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
