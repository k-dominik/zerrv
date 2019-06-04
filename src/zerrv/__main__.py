from argparse import ArgumentParser, Action
from PIL import Image
from zerrv.external.array5d import Array5D, Point5D, Chunk5D, Blocking5D
from zerrv.external.data_source import H5N5DataSource
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


from zerrv.util import h5n5_file, handle_path, H5N5JsonEncoder


logger = logging.getLogger(__name__)

app = Starlette(debug=True)


class MyJSONResponse(JSONResponse):
    """Response class that uses custom JSON encoder"""

    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"), cls=H5N5JsonEncoder
        ).encode("utf-8")


@app.route("/{path:path}/tilelayer/{level}/{t:int}/{c:int}/{z:int}/{y:int}/{x:int}.png", methods=["GET"])
async def tilelayer(request):
    """Endpoint to serve a single channel tile layer to openlayers

    URL params:
      level: zoom-level, ignored atm.
      t: time-point -> not blocked
      c: channel -> not blocked
      z: z-slice in pixel/voxel coordinates -> not blocked
      y: _block_ number in y-direction
      x: _block_ number in x-direction
    """
    logger.debug(f"doing tilelayer request at {request.path_params['path']}")
    logger.debug(f"requesting tile (tczyx): {[request.path_params[a] for a in 'tczyx']}")
    p = request["app"].serve_rootpath / request.path_params["path"]

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
    p = request["app"].serve_rootpath / request.path_params["path"]

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
    p = request["app"].serve_rootpath / request.path_params["path"]
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


def parse_args():
    class ExistingPathAction(Action):
        def __init__(self, option_strings, dest, type=pathlib.Path, **kwargs):
            super().__init__(option_strings, dest, **kwargs)
            assert type == pathlib.Path

        def __call__(self, parser, namespace, values, option_string=None):
            assert isinstance(values, str), "only one value may be supplied here"
            p = pathlib.Path(values)
            if not p.exists():
                raise IOError(f"Path {p} does not exist!")
            p = p.absolute()
            setattr(namespace, self.dest, p)

    p = ArgumentParser()

    p.add_argument("--host", default="0.0.0.0", help="host address")
    p.add_argument("--port", default=8978, type=int, help="port number")
    p.add_argument(
        "rootpath", type=pathlib.Path, action=ExistingPathAction, help="path from which to start serving files."
    )

    args = p.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    logger.info(f"serving directory {args.rootpath} from http://{args.host}:{args.port}")
    app.serve_rootpath = args.rootpath
    uvicorn.run(app, host=args.host, port=args.port, loop="asyncio")


if __name__ == "__main__":
    main()
