import errno
from pathlib import Path


def ensure_new_output_path(value):
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(errno.EEXIST, "Output path already exists", str(path))
    for parent in path.parents:
        if parent.exists() or parent.is_symlink():
            if not parent.is_dir():
                raise NotADirectoryError(errno.ENOTDIR, "Output path parent is not a directory", str(parent))
            break
    return path
