from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import Response, FileResponse, StreamingResponse
import uvicorn
import z5py
import numpy
from PIL import Image
import io
import shutil


from dataclasses import dataclass


app = Starlette(debug=True)


@dataclass
class BlockCoordinates:
    axisorder: str
    shape: numpy.array

    def get_slicing(self, block_coords: tuple, axisorder: str):
        return (slice(256, 384), slice(256, 384), slice(0, 1))


@app.route("/{filename}/tilelayer/{t}/{c}/{z}/{y}/{x}.png", methods=["GET"])
async def tilelayer(request):
    z5file = z5py.File(request.path_params["filename"], "r")
    axisorder = "xyc"
    bc = BlockCoordinates(axisorder=axisorder, shape=z5file["data"].shape)
    block_coords = tuple(request.path_params[k] for k in axisorder)
    slicing = bc.get_slicing(block_coords=block_coords, axisorder=axisorder)
    data = z5file["data"][slicing]
    img = Image.fromarray(data.squeeze(), mode="P")
    bo = io.BytesIO()
    img.save(bo, format="PNG")
    bo.seek(0)
    return StreamingResponse(bo, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
