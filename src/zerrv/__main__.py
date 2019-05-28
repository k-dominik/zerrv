from PIL import Image
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import Response, FileResponse, StreamingResponse
from starlette.requests import Request
from starlette.convertors import Convertor
import h5py
import io
import json
import logging
import numpy
import shutil
import pathlib
import uvicorn
import z5py
import typing


from dataclasses import dataclass


logger = logging.getLogger(__name__)

app = Starlette(debug=True)


class H5N5JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (z5py.group.Group, h5py.Group)):
            return {"name": obj.name, "type": "group"}
        if isinstance(obj, (z5py.dataset.Dataset, h5py.Dataset)):
            # HACK
            return {"name": obj.path.split("/")[-1], "type": "dataset"}
        else:
            json.JSONEncoder.default(self, obj)


class MyJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=H5N5JsonEncoder,
        ).encode("utf-8")


@dataclass
class BlockCoordinates:
    axisorder: str
    shape: numpy.array
    block_size: numpy.array

    def get_normalized_slicing(self, t=0, c=0, z=0, y=0, x=0):
        logger.debug(f"get_normalized_slicing: {t}{c}{z}{y}{x}")
        block_size = self.block_size
        return (
            slice(t * block_size[0], (t + 1) * block_size[0]),
            slice(c * block_size[1], (c + 1) * block_size[1]),
            slice(z * block_size[2], (z + 1) * block_size[2]),
            slice(y * block_size[3], (y + 1) * block_size[3]),
            slice(x * block_size[4], (x + 1) * block_size[4]),
        )

    def get_slicing(self, t=0, c=0, z=0, y=0, x=0):
        normalized_slicing = self.get_normalized_slicing(t, c, z, y, x)
        inds = ["tczyx".index(a) for a in self.axisorder]
        logger.debug(
            f"inds: {inds} - {normalized_slicing} -> {[normalized_slicing[i] for i in inds]}"
        )
        return tuple(normalized_slicing[i] for i in inds)


def handle_path(path: pathlib.Path):
    pathstr = str(path)
    logger.debug(f"pathstr: {pathstr}")
    if any(p.endswith(".n5") for p in path.parts):
        *external, internal = pathstr.split(".n5")
        logger.debug(f"external: {external}, internal: {internal}")
        external_path = "".join(external) + ".n5"
        internal.lstrip("/")
        return pathlib.Path(external_path), internal, True
    elif path.match("*.h5"):
        raise NotImplementedError()
    else:
        return path, "", False


def h5n5_file(filename):
    if filename.match("*.n5"):
        return z5py.File(filename, "r")
    elif filename.match("*.h5"):
        return h5py.File(filename, "r")


class H5N5Dataset:
    def __init__(self, filename, internal_path=""):
        self._internal_path = internal_path
        self._file = h5n5_file(filename)
        self._axistags = "zyxc"

    @property
    def axistags(self):
        logger.warning("not really implemented... yo")
        return self._axistags

    @property
    def normalized_axistags(self):
        return "tczyx"

    @property
    def normalized_shape(self):
        inds = [self.normalized_axistags.index(x) for x in self.axistags]
        new_shape = [0, 0, 0, 0, 0]
        for ind, extent in zip(inds, self.raw().shape):
            new_shape[ind] = extent

        return tuple(new_shape)

    @property
    def normalized_chunk_shape(self):
        inds = [self.normalized_axistags.index(x) for x in self.axistags]
        new_chunk_shape_shape = [0, 0, 0, 0, 0]
        # for ind, extent in zip(inds, self.raw().chunks):
        #     new_chunk_shape_shape[ind] = extent
        new_chunk_shape_shape = [0, 1, 1, 128, 128]
        return tuple(new_chunk_shape_shape)

    def raw(self):
        return self._file[self._internal_path]


@app.route(
    "/{path:path}/tilelayer/{level}/{t:int}/{c:int}/{z:int}/{y:int}/{x:int}.png",
    methods=["GET"],
)
async def tilelayer(request):
    logger.debug(f"doing tilelayer request at {request.path_params['path']}")
    logger.debug(
        f"requesting tile (tczyx): {[request.path_params[a] for a in 'tczyx']}"
    )
    p = pathlib.Path(request.path_params["path"])

    external, internal, is_h5_n5 = handle_path(p)
    logger.debug(f"external: {external}, internal: {internal}, is_h5_n5: {is_h5_n5}")

    if not external.exists():
        return JSONResponse(
            {"external": str(external), "internal": str(internal), "b": is_h5_n5}
        )
    if not is_h5_n5:
        return JSONResponse(
            {"external": str(external), "internal": str(internal), "b": is_h5_n5}
        )
    z5dataset = H5N5Dataset(external, internal)
    axisorder = z5dataset.axistags
    bc = BlockCoordinates(
        axisorder=z5dataset.axistags,
        shape=z5dataset.normalized_shape,
        block_size=z5dataset.normalized_chunk_shape,
    )
    block_coords = {k: request.path_params[k] for k in z5dataset.normalized_axistags}
    slicing = bc.get_slicing(**block_coords)
    logger.debug(
        f"getting data with slicing: {slicing} - shape: {z5dataset.raw().shape}"
    )
    data = z5dataset.raw()[slicing]
    logger.debug(f"data.shape: {data.shape}")
    img = Image.fromarray(data.squeeze(), mode="P")
    bo = io.BytesIO()
    img.save(bo, format="PNG")
    bo.seek(0)
    return StreamingResponse(bo, media_type="image/png")


@app.route("/{path:path}/info.json", methods=["GET"])
async def info(request):
    logger.debug(f"doing info request at {request.path_params['path']}")
    p = pathlib.Path(request.path_params["path"])

    external, internal, is_h5_n5 = handle_path(p)
    logger.debug(f"external: {external}, internal: {internal}, is_h5_n5: {is_h5_n5}")

    if not external.exists():
        return JSONResponse(
            {"external": str(external), "internal": str(internal), "b": is_h5_n5}
        )
    if not is_h5_n5:
        return JSONResponse(
            {"external": str(external), "internal": str(internal), "b": is_h5_n5}
        )

    logger.debug(f"trying to open file")
    h5n5_dataset = H5N5Dataset(external, internal)
    d = h5n5_dataset.raw()
    assert isinstance(d, (z5py.dataset.Dataset, h5py.Dataset))

    x_str = "{" + "}/{".join(h5n5_dataset.normalized_axistags) + "}.png"
    return JSONResponse(
        {
            "shape": h5n5_dataset.normalized_shape,
            "block_size": h5n5_dataset.normalized_chunk_shape,
            "url": f"{external}/{internal}/{x_str}",
        }
    )


@app.route("/{path:path}", methods=["GET"])
async def query(request):
    logger.debug(f"doing parse_request at {request.path_params['path']}")
    p = pathlib.Path(request.path_params["path"])
    print(f"{p}, {type(p)}")
    if not p.exists():
        return {}

    external, internal, is_h5_n5 = handle_path(p)

    def get_type(p):
        if p.match("*.n5"):
            return "n5-file"
        elif p.match("*.h5"):
            return "h5-file"
        else:
            if p.is_dir():
                return "folder"
            else:
                return "unknown"

    if not is_h5_n5:
        files = [*p.glob("*.h5"), *p.glob("*.n5")]
        folders = [x for x in p.iterdir() if x.is_dir() and x not in files]
        ret_dict = [
            *[{"name": str(f), "type": get_type(f)} for f in files],
            *[{"name": str(f), "type": get_type(f)} for f in folders],
        ]
    else:
        f = h5n5_file(external)
        if internal:
            g = f[internal]
            assert isinstance(g, (z5py.group.Group, h5py.Group))
        else:
            g = f
        ret_dict = list(g.values())

    return MyJSONResponse(ret_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
