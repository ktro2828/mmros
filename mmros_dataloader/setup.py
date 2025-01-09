from __future__ import annotations

import os.path as osp
from glob import glob

from setuptools import find_packages, setup

package_name = "mmros_dataloader"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (osp.join("share", package_name), ["package.xml"]),
        (osp.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ktro2828",
    maintainer_email="kotaro.uetake@tier4.jp",
    description="The ROS 2 package to load and publish open dataset.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"{package_name}_nuscenes_exe = {package_name}.nuscenes:main",
            f"{package_name}_nuimages_exe = {package_name}.nuimages:main",
        ],
    },
)
