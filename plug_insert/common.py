from pathlib import Path


def project_path() -> Path:
    return Path("~/.mohou/plug_insert").expanduser()
