"""AI Fraud Detector - REST API + dashboard host.

    python 4_Backend/api/fraud_detection_api.py            # http://127.0.0.1:5000
    python 4_Backend/api/fraud_detection_api.py --seed     # load demo orders first (--force if reviews exist)

Set FRAUD_API_KEY to require an `X-API-Key` header on the write routes (see risk_score_endpoint.py).

GET /                  merchant dashboard (5_Frontend/dashboard)
GET /api/health        liveness + loaded model version and quality metrics
... risk routes in risk_score_endpoint.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "4_Backend" / "api", ROOT / "4_Backend" / "database", ROOT / "4_Backend" / "utils", ROOT / "3_Model"):
    sys.path.insert(0, str(p))

from flask import Flask, jsonify, send_from_directory  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from connection import DB_PATH, ReviewsExist, init_db, seed_demo_data  # noqa: E402
from preprocessing import FEATURES  # noqa: E402
from risk_score_endpoint import bp as risk_bp  # noqa: E402
from risk_scoring import RiskScorer  # noqa: E402

log = logging.getLogger("fraud_api")

FRONTEND = ROOT / "5_Frontend"


def create_app(db_path: Path = DB_PATH) -> Flask:
    app = Flask(__name__, static_folder=None)
    # refuses to start if the saved model was trained on a different feature list than the code builds
    app.config.update(DB_PATH=db_path, SCORER=RiskScorer(expected_features=FEATURES), JSON_SORT_KEYS=False)
    init_db(db_path)
    app.register_blueprint(risk_bp)

    @app.get("/api/health")
    def health():
        s = app.config["SCORER"]
        return jsonify(status="ok", model_version=s.version,
                       thresholds={"review": s.review_threshold, "block": s.block_threshold},
                       metrics={k: s.metrics.get(k) for k in ("roc_auc", "pr_auc")},
                       auth_required=bool(os.getenv("FRAUD_API_KEY")))

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND / "dashboard", "index.html")

    @app.get("/dashboard/<path:name>")
    def dashboard_files(name):
        return send_from_directory(FRONTEND / "dashboard", name)

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(FRONTEND / "assets", name)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(HTTPException)
    def http_error(err):  # 400 / 405 / 413 ... as JSON, like the rest of the API
        return jsonify(error=err.description or err.name), err.code

    @app.errorhandler(Exception)
    def server_error(err):  # never an HTML stack-trace page; the traceback goes to the server log
        log.exception("unhandled error")
        return jsonify(error="internal server error"), 500

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="(re)load demo orders from the held-out test set")
    ap.add_argument("--force", action="store_true", help="with --seed: replace orders even if reviews exist (backed up first)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    if a.seed or not DB_PATH.exists():
        try:
            print(f"seeding demo data: {seed_demo_data(force=a.force)} orders")
        except ReviewsExist as e:
            sys.exit(f"refusing to seed: {e}")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not os.getenv("FRAUD_API_KEY"):
        print(f"WARNING: listening on {a.host} with no FRAUD_API_KEY - anyone who can reach this port can "
              "score, review and read orders. Set FRAUD_API_KEY to require an X-API-Key header.", file=sys.stderr)
    create_app().run(host=a.host, port=a.port, debug=False)
