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

    if args.search:
        from src.gridsearch import run_search
        return run_search(args.config, args.data_dir, args.output_dir, logger,
                          save=args.save)

    if cfg.task == "anomaly_detection":
        from src.pipelines import anomaly
        return anomaly.run(cfg, Path(args.config), args.data_dir,
                           args.output_dir, save=args.save, logger=logger)
    if cfg.task == "forecasting":
        from src.pipelines import forecast
        return forecast.run(cfg, Path(args.config), args.data_dir,
                            args.output_dir, save=args.save, logger=logger)
    if cfg.task == "regime_clustering":
        from src.pipelines import clustering
        return clustering.run(cfg, Path(args.config), args.data_dir,
                              args.output_dir, save=args.save, logger=logger)
    raise ValueError(cfg.task)


def _dispatch_predict(args):
    logger = setup_logger(args.verbose)
    run_meta = load_json(Path(args.run_dir) / "run_metadata.json")
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(run_meta["config"], f); tmp = Path(f.name)
    try:
        cfg = Config.load(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    if cfg.task == "anomaly_detection":
        from src.pipelines import anomaly
        return anomaly.run(cfg, Path(run_meta["config_file"]),
                           args.data_dir, args.output_dir,
                           save=args.save, logger=logger)
    if cfg.task == "forecasting":
        from src.pipelines import forecast
        return forecast.run(cfg, Path(run_meta["config_file"]),
                            args.data_dir, args.output_dir,
                            save=args.save, logger=logger)
    if cfg.task == "regime_clustering":
        from src.pipelines import clustering
        return clustering.run(cfg, Path(run_meta["config_file"]),
                              args.data_dir, args.output_dir,
                              save=args.save, logger=logger)
    raise ValueError(cfg.task)


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
    t.add_argument("--search", action="store_true",
                   help="Ejecuta grid-search usando el bloque 'search' del JSON")
    t.set_defaults(func=_dispatch_train)

    # ---- predict ----
    pr = sub.add_parser("predict", help="Reutilizar un modelo previamente guardado")
    pr.add_argument("--run-dir", type=Path, required=True,
                    help="outputs/<csv>/<config>/run_XXXX")
    pr.add_argument("--data-dir", type=Path, default=Path("data"))
    pr.add_argument("--output-dir", type=Path, default=Path("outputs"))
    pr.add_argument("--save", action="store_true")
    pr.set_defaults(func=_dispatch_predict)

    s = sub.add_parser("search", help="Grid-search directo sobre un JSON")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--data-dir", type=Path, default=Path("data"))
    s.add_argument("--output-dir", type=Path, default=Path("outputs"))
    s.set_defaults(func=lambda a: __import__("src.gridsearch", fromlist=["run_search"])
                   .run_search(a.config, a.data_dir, a.output_dir,
                               setup_logger(a.verbose), save=True))

    bs = sub.add_parser("batch", help="Ejecuta todos los JSON de una carpeta")
    bs.add_argument("--configs-dir", type=Path, required=True)
    bs.add_argument("--data-dir", type=Path, default=Path("data"))
    bs.add_argument("--output-dir", type=Path, default=Path("outputs"))
    bs.add_argument("--save", action="store_true")
    bs.set_defaults(func=_dispatch_batch)

    return p


def _dispatch_batch(args):
    from src.gridsearch import run_search
    logger = setup_logger(args.verbose)
    json_files = sorted(args.configs_dir.glob("*.json"))
    logger.info(f"[batch] {len(json_files)} configs en {args.configs_dir}")
    results = []
    for jf in json_files:
        try:
            cfg = Config.load(jf)
            if args.search and cfg.search:
                results.append({jf.name: run_search(jf, args.data_dir,
                                                    args.output_dir, logger,
                                                    save=args.save)})
            else:
                results.append({jf.name: _dispatch_train(
                    type("A", (), {"config": jf, "verbose": args.verbose,
                                   "data_dir": args.data_dir,
                                   "output_dir": args.output_dir,
                                   "save": args.save, "search": False})())})
        except Exception as e:
            logger.warning(f"[batch] {jf.name} falló: {e}")
            results.append({jf.name: {"error": str(e)}})
    return results

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())