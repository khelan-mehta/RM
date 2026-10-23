"""Launch the Khelan server.

Local (default):   python khelan/run.py
                   → http://127.0.0.1:8000  (localhost only, not exposed)

Remote / LAN:      python khelan/run.py --remote
                   → https://<your-lan-ip>:8443  (self-signed TLS, auth required)
                     reachable from your phone on the same Wi-Fi.

Env: KHELAN_PORT, KHELAN_HOST, KHELAN_MIC, KHELAN_PASSWORD, KHELAN_REMOTE=1.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import uvicorn

if __name__ == "__main__":
    remote = "--remote" in sys.argv or os.environ.get("KHELAN_REMOTE") == "1"
    kwargs = dict(log_level="warning")
    if remote:
        from certs import ensure_cert, lan_ips
        cert, key = ensure_cert()
        host = "0.0.0.0"
        port = int(os.environ.get("KHELAN_PORT", "8443"))
        kwargs.update(ssl_certfile=cert, ssl_keyfile=key)
        print("\nKhelan remote (HTTPS) — reachable on this network at:")
        for ip in lan_ips() or ["<this-machine-ip>"]:
            print(f"    https://{ip}:{port}")
        print("  Accept the self-signed certificate warning once, then sign in.\n")
    else:
        host = os.environ.get("KHELAN_HOST", "127.0.0.1")
        port = int(os.environ.get("KHELAN_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, **kwargs)
