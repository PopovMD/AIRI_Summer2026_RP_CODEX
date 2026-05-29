#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import subprocess
import sys


CODEX_REPO = "https://github.com/sschrod/CODEX.git"


# Запускает внешнюю команду и завершает скрипт при ошибке
def sh(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


# Проверяет версии ключевых пакетов перед запуском 
def check_environment() -> None:
    """Fail early on the package conflict that caused the Scanpy/Zarr/NumPy errors."""
    import numpy
    import scipy
    import scanpy
    import zarr
    import numcodecs
    from gears import PertData  # noqa: F401

    print("Python:", sys.version.split()[0])
    print("numpy:", numpy.__version__)
    print("scipy:", scipy.__version__)
    print("scanpy:", scanpy.__version__)
    print("zarr:", zarr.__version__)
    print("numcodecs:", numcodecs.__version__)

    # фиксим конфликт нумпая и scanpy
    if numpy.__version__.startswith("2."):
        raise RuntimeError(
            "NumPy 2.x is installed. For this CODEX/GEARS/Scanpy setup use numpy==1.26.4."
        )

    # фиксим конфликты zarr и numcodecs
    if zarr.__version__.startswith("2.") and tuple(map(int, numcodecs.__version__.split(".")[:2])) >= (0, 16):
        raise RuntimeError(
            "Incompatible zarr/numcodecs pair. Use zarr==2.18.7 and numcodecs<0.16."
        )


# Клонирует CODEX
def clone_codex_if_needed(codex_dir: pathlib.Path) -> None:

    if (codex_dir / "CODEX_gene_perturbation.py").exists():
        print(f"Using CODEX repo: {codex_dir}")
        return

    # пустая папка =>
    if codex_dir.exists() and any(codex_dir.iterdir()):
        raise FileNotFoundError(
            f"{codex_dir} exists but does not contain CODEX_gene_perturbation.py. "
            "Pass --codex-dir pointing to a CODEX clone or use an empty path."
        )

    #скачивает CODEX из GitHub
    codex_dir.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", CODEX_REPO, str(codex_dir)])


# Патчит установленный GEARS для совместимости с AnnData/Pandas.
def patch_gears_indexing() -> None:
    """Patch installed GEARS, not CODEX. This avoids an AnnData/Pandas indexing issue."""
    import gears.pertdata

    # Находит путь к установленному файлу GEARS pertdata.py.
    path = pathlib.Path(inspect.getfile(gears.pertdata))

    old = "self.adata = self.adata[filter_go.index.values, :]"
    new = "self.adata = self.adata[np.array(filter_go.index.values), :]"
    text = path.read_text()

    if new in text:
        print(f"GEARS patch already present: {path}")
        return

    if old not in text:
        print(f"GEARS patch target not found, skipping: {path}")
        return

    # исправленная версия файла GEARS
    path.write_text(text.replace(old, new))
    print(f"Patched GEARS: {path}")


# Собирает команду запуска оригинального CODEX-скрипта
def build_codex_command(args: argparse.Namespace) -> list[str]:
    # Подготавливает абсолютные пути к данным и результатам
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Формирует команду для CODEX_gene_perturbation.py
    cmd = [
        sys.executable,
        "-u",
        "CODEX_gene_perturbation.py",
        "-dn",
        args.dataset,
        "-dp",
        str(data_dir) + os.sep,
        "--save_folder",
        str(results_dir) + os.sep,
        "-l",
        *map(str, args.layers),
        "-s",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "-p",
        str(args.patience),
        "-bs",
        str(args.batch_size),
    ]

    if args.download_data:
        cmd.append("--download_data")

    # Пробрасывает дополнительные аргументы напрямую в CODEX
    if args.extra:
        cmd.extend(args.extra)

    return cmd


# Считывает аргументы командной строки launcher-скрипта
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thin launcher for the original CODEX repository entrypoint."
    )

    # Пути
    p.add_argument("--codex-dir", type=pathlib.Path, default=pathlib.Path("CODEX"))
    p.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    p.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))

    # Настройки датасета
    p.add_argument("--dataset", default="norman")
    p.add_argument("--download-data", action="store_true")

    # параметры обучения
    p.add_argument("--layers", nargs="+", type=int, default=[256, 64, 32])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)

    #пропустить проверку окружения
    p.add_argument("--skip-env-check", action="store_true")

    # Принимает доп аргументы после разделителя 
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed to CODEX_gene_perturbation.py after --, e.g. -- --predict_and_eval",
    )
    return p.parse_args()


# Основной сценарий: проверка, подготовка и запуск CODEX
def main() -> None:
    args = parse_args()

    # Удаляет разделитель перед аргументами
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    # Проверяет окружение
    if not args.skip_env_check:
        check_environment()

    # Готовит локальный CODEX и патчит GEARS
    clone_codex_if_needed(args.codex_dir)
    patch_gears_indexing()

   # запуск
    cmd = build_codex_command(args)
    sh(cmd, cwd=args.codex_dir.resolve())


if __name__ == "__main__":
    main()
