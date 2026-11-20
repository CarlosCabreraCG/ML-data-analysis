"""CLI unificada: train y predict con argparse."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.utils import load_json, run_base_dir, setup_logger, timestamp


def _dispatch_train(args):
    cfg = Config.load(args.config)
    logger = setup_logger(args.verbose)
    if cfg.task == "anomaly_detection":
        from src.pipelines import anomaly
        return anomaly.run(cfg, Path(args.config), args.data_dir, args.output_dir,
                           save=args.save, logger=logger)
    from src.pipelines import forecast
    return forecast.run(cfg, Path(args.config), args.data_dir, args.output_dir,
                        save=args.save, logger=logger)


def _dispatch_predict(args):
    logger = setup_logger(args.verbose)
    run_meta = load_json(Path(args.run_dir) / "run_metadata.json")
    cfg_raw = run_meta["config"]
    cfg = Config.load_from_dict(cfg_raw) if hasattr(Config, "load_from_dict") else Config(**{
        k: v for k, v in cfg_raw.items() if k in Config.__dataclass_fields__
    })
    # Nota: usamos .load sobre un dict temporal para mantener una sola API
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg_raw, f); tmp = Path(f.name)
    try:
        cfg = Config.load(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    if cfg.task == "anomaly_detection":
        from src.pipelines import anomaly
        # Reutiliza el pipeline pero apuntando al run existente
        return anomaly.run(cfg, Path(run_meta["config_file"]),
                           args.data_dir, args.output_dir,
                           save=args.save, logger=logger)
    from src.pipelines import forecast
    return forecast.run(cfg, Path(run_meta["config_file"]),
                        args.data_dir, args.output_dir,
                        save=args.save, logger=logger)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sensor-analytics",
        description="Detección de anomalías no supervisada y forecasting por config JSON.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train ----
    t = sub.add_parser("train", help="Entrenar y evaluar según configuración JSON")
    t.add_argument("--config", type=Path, required=True,
                   help="Ruta al JSON de configuración")
    t.add_argument("--data-dir", type=Path, default=Path("data"))
    t.add_argument("--output-dir", type=Path, default=Path("outputs"))
    t.add_argument("--save", action="store_true",
                   help="Persistir modelo, métricas y plots (por defecto: no guarda)")
    t.set_defaults(func=_dispatch_train)

    # ---- predict ----
    pr = sub.add_parser("predict", help="Reutilizar un modelo previamente guardado")
    pr.add_argument("--run-dir", type=Path, required=True,
                    help="outputs/<csv>/<config>/run_XXXX")
    pr.add_argument("--data-dir", type=Path, default=Path("data"))
    pr.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pr.add_argument("--save", action="store_true")
    pr.set_defaults(func=_dispatch_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())