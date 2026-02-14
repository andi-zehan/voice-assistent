"""Leonardo Voice Assistant — Server entry point.

Loads config, initializes components, and starts the uvicorn server.
"""

import logging
import sys
from pathlib import Path

import yaml

from server.app import create_app


def load_config(path: str | None = None) -> dict:
    """Load server config from YAML file."""
    if path is None:
        # Default: server/config.yaml relative to this file
        path = str(Path(__file__).parent / "config.yaml")

    config_path = Path(path)
    if not config_path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Leonardo Voice Assistant Server")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--host", type=str, default=None, help="Bind host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8765)

    app = create_app(config)

    import uvicorn
    print(f"\033[32mLeonardo server starting on ws://{host}:{port}/ws\033[0m")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
