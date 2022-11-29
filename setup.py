from setuptools import setup

setup_requires = []

install_requires = []

setup(
    name="plug_insert",
    version="0.0.1",
    description="plug insert",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    package_data={"skplan": ["py.typed"]},
)
