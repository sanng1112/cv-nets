import importlib
import os
from typing import Sequence

from common import LIBRARY_ROOT
from utils import logger


def import_modules_from_folder(
    folder_name: str, extra_roots: Sequence[str] = ()
) -> None:
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        logger.error(
            f"{folder_name} doesn't exist in the public library root directory."
        )

    for base_dir in [".", *extra_roots]:
        for path in LIBRARY_ROOT.glob(os.path.join(base_dir, folder_name, "**/*.py")):
            filename = path.name
            if filename[0] not in (".", "_"):
                module_name = str(
                    path.relative_to(LIBRARY_ROOT).with_suffix("")
                ).replace(os.sep, ".")
                importlib.import_module(module_name)
