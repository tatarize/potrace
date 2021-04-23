# Python potrace
Pure Python Port of Potrace. This is a python port of Peter Selinger's Potrace (based on 1.16 code).

Rather than using the original bitmap code this port uses the python Pillow library for graphics loading, and editing.

This port is needed because many python hooks to the original code such as `pypotrace` have installation issues with some OSes, and compilation problems.

# Installing

The intent is to permit pip installation, however this is not yet an approved port, and cannot be uploaded to pypi until it is. The intent is to use the `potrace` namespace on pypi: https://pypi.org/project/potrace/


For now, download the download the package library and type: `pip install -U .` in the package directory. Addon backends will be automatically detected if installed with pip. `backend-svg` for how to author a plugin backend. 


```
usage: potrace [-h] [-v] [-l] [-o OUTPUT] [-b {svg,jagged-svg}]
               [-z {black,white,left,right,minority,majority,random}]
               [-t TURDSIZE] [-a ALPHAMAX] [-n] [-O OPTTOLERANCE] [-C COLOR]
               [-i] [-k BLACKLEVEL] [-s SCALE] [-1]
               [filename]

positional arguments:
  filename              an input file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         prints version info and exit
  -l, --license         prints license info and exit
  -o OUTPUT, --output OUTPUT
                        write all output to this file
  -b {svg,jagged-svg}, --backend {svg,jagged-svg}
                        select backend by name
  -z {black,white,left,right,minority,majority,random}, --turnpolicy {black,white,left,right,minority,majority,random}
                        how to resolve ambiguities in path decomposition
  -t TURDSIZE, --turdsize TURDSIZE
                        suppress speckles of up to this size (default 2)
  -a ALPHAMAX, --alphamax ALPHAMAX
                        corner threshold parameter
  -n, --longcurve       turn off curve optimization
  -O OPTTOLERANCE, --opttolerance OPTTOLERANCE
                        curve optimization tolerance
  -C COLOR, --color COLOR
                        set foreground color (default Black)
  -i, --invert          invert bitmap
  -k BLACKLEVEL, --blacklevel BLACKLEVEL
                        invert bitmap
  -s SCALE, --scale SCALE
                        Scale the image by an integer factor n>0.
  -1, --dither          Dither rather than threshold to 1-bit.
```

# Requirements
* PIL/Pillow is required for image loading and modifications.

# License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2, or (at your option) any later version.

Furthermore, this is permitted to be relicensed under any terms the Peter Selinger's original Potrace is licensed under. If he broadly publishes the software under a more permissive license this port should be considered licensed as such as well. Further, if you purchase a proprietary license for inclusion within commercial software under his Dual Licensing program your use of this software shall be under whatever terms he permits for that. Any contributions to this port must be made under equally permissive terms.

"Potrace" is a trademark of Peter Selinger.
