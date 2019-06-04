from typing import Union, Tuple
import h5py
import json
import pathlib
import z5py


import logging

logger = logging.getLogger(__name__)


class H5N5JsonEncoder(json.JSONEncoder):
    """JSON encoder for h5/n5 dataset/group objects

    Used to allow for a "file-like" browse access.
    """

    def default(self, obj):
        if isinstance(obj, (z5py.group.Group, h5py.Group)):
            return {"name": obj.name, "type": "group"}
        if isinstance(obj, (z5py.dataset.Dataset, h5py.Dataset)):
            # HACK
            return {"name": obj.path.split("/")[-1], "type": "dataset"}
        else:
            json.JSONEncoder.default(self, obj)


def h5n5_file(filename: str) -> Union[h5py.File, z5py.File]:
    if filename.match("*.n5"):
        return z5py.File(filename, "r")

    elif filename.match("*.h5"):
        return h5py.File(filename, "r")


def handle_path(path: pathlib.Path) -> Tuple[pathlib.Path, str, bool]:
    """Split path into external and internal path components

    Args:
        path: Full path to dataset/group

    Returns:
        Tuple with external_path, internal_path, is_h5n5
    """
    pathstr = str(path)
    logger.debug(f"handling path: {pathstr}")
    if any(p.endswith(".n5") for p in path.parts):
        *external_path, internal_path = pathstr.split(".n5")
        logger.debug(f"external_path: {external_path}, internal_path: {internal_path}")
        external_path = "".join(external_path) + ".n5"
        internal_path.lstrip("/")
        return pathlib.Path(external_path), internal_path, True
    elif path.match("*.h5"):
        raise NotImplementedError()
    else:
        return path, "", False
