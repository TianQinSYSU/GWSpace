import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

gsl_prefix = Path(os.environ.get("GSL_PREFIX", os.environ.get("CONDA_PREFIX", "/usr")))

include_dirs = ["include", np.get_include(), str(gsl_prefix / "include")]
library_dirs = [str(gsl_prefix / "lib")]
libraries = ["gsl", "gslcblas", "m"]

extensions = [
    Extension(
        "gwspace.libFastGB",
        sources=[
            "src/FastGB.pyx",
            "src/spacecrafts.c",
            "src/GB.c",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=["-std=c99", "-O3"],
    ),
    Extension(
        "gwspace.pyIMRPhenomD",
        sources=[
            "src/pyIMRPhenomD.pyx",
            "src/IMRPhenomD.c",
            "src/IMRPhenomD_internals.c",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=["-std=c99", "-O3"],
    ),
]

# translate the constants.h to constants.py
fp_const_h = "./include/constants.h"
fp_const_py = "./gwspace/constants.py"

with open(fp_const_h, "r") as fp_in:
    with open(fp_const_py, "w") as fp_out:
        lines = fp_in.readlines()
        for line in lines:
            if (len(line.split())) >= 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except ValueError as e:
                        continue

setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={"language_level": "3"},)
)
