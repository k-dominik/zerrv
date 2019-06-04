from PIL import Image
from external.array5d import Array5D, Point5D, Chunk5D, Blocking5D
from external.data_source import H5N5DataSource
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
            content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"), cls=H5N5JsonEncoder
        ).encode("utf-8")


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


@app.route("/{path:path}/tilelayer/{level}/{t:int}/{c:int}/{z:int}/{y:int}/{x:int}.png", methods=["GET"])
async def tilelayer(request):
    logger.debug(f"doing tilelayer request at {request.path_params['path']}")
    logger.debug(f"requesting tile (tczyx): {[request.path_params[a] for a in 'tczyx']}")
    p = pathlib.Path(request.path_params["path"])

    external_path, internal_path, is_h5_n5 = handle_path(p)
    logger.debug(f"external_path: {external_path}, internal_path: {internal_path}, is_h5_n5: {is_h5_n5}")

    if not external_path.exists():
        return JSONResponse({"external_path": str(external_path), "internal_path": str(internal_path), "b": is_h5_n5})

    if not is_h5_n5:
        return JSONResponse({"external_path": str(external_path), "internal_path": str(internal_path), "b": is_h5_n5})

    z5dataset = H5N5DataSource(f"{external_path}{internal_path}")
    tile_shape = z5dataset.tile_shape()
    tile_shape = Chunk5D(x=tile_shape.x, y=tile_shape.y, z=1, c=1, t=1)
    bc = Blocking5D(tile_shape)
    block_coords = Point5D(**{k: request.path_params[k] for k in "tczyx"})
    slicing = bc.get_slice(block_coords)
    logger.debug(f"getting data with slicing: {slicing} - shape: {z5dataset.shape}")
    data = z5dataset.do_retrieve(slicing)
    logger.debug(f"data.shape: {data.shape}")
    img = Image.fromarray(data.raw("yx"), mode="P")
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
        return JSONResponse({"external": str(external), "internal": str(internal), "b": is_h5_n5})
    if not is_h5_n5:
        return JSONResponse({"external": str(external), "internal": str(internal), "b": is_h5_n5})

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
