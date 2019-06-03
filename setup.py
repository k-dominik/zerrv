from setuptools import setup, find_packages

setup(
    name="zerrv",
    version="0.1.0dev1",
    author="Dominik Kutra",
    author_email="author email address",
    license="MIT",
    description="http server to expose zarr/n5/hdf5 volumes",
    # long_description=description,
    # url='https://...',
    package_dir={"": "src"},
    packages=find_packages("./src"),
    include_package_data=True,
    install_requires=["aiofiles", "h5py", "numpy", "starlette", "uvicorn" "z5py"],
    entry_points={"console_scripts": ["zerrv = zerrv.__main__:main"]},
)
