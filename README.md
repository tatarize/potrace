# potrace
Pure Python Port of Potrace. This is intended to be a pure python port of Peter Selinger's Potrace (based on 1.16 code). However, rather than using the original bitmap code we will use the python Pillow library for graphics loading. The intent is that backends provided within Potrace will also be made availible here as well as the option entrypoint plugin backends. This may only end up by svg with optional plugin code.

This port is generally because most python hooks to the original code such as `pypotrace` have installation issues with some of the stuff breaking and not compiling. Also, I thought I could hammer this out in a day after seeing code like https://github.com/kilobtye/potrace javascript port but, since that's implemented licensed in GPL and I wanted this licensed in `GPL2+ and however Peter Selinger's code is licensed`. I needed to port the more feature-rich C code which exceeded my original time estimates.

# Current progress
* Currently some decomposition with the relevant options with some bugs. Some work still needs to be done for the port.
* The second phase of line and curve conversion does not yet work.

# Installing
As this currently does not fully work, installing is not recommended and only this package is availible.

The intent, however, is to permit a python entrypoint hook to run the script as one would run original potrace with similar command line arguments.

```
usage: potrace.py [-h] [-v] [-l] [-o OUTPUT] [-b {svg}]
                  [-z {black,white,left,right,minority,majority,random}]
                  [-P PROFILE] [-t TURDSIZE] [-a ALPHAMAX] [-n]
                  [-O OPTTOLERANCE] [-u UNIT] [-C COLOR] [-g] [-f]
                  [filename]

positional arguments:
  filename              an input file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         prints version info and exit
  -l, --license         prints license info and exit
  -o OUTPUT, --output OUTPUT
                        write all output to this file
  -b {svg}, --backend {svg}
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
  -u UNIT, --unit UNIT  quantize output to 1/unit pixels
  -C COLOR, --color COLOR
                        set foreground color (default Black)
  -g, --group           group related paths together
  -f, --flat            whole image as a single path
```

# Requirements
* PIL/Pillow is required for image loading and modifications.

# License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2, or (at your option) any later version.

Furthermore, this is permitted to be relicensed under any terms the Peter Selinger's original Potrace is licensed under. If he broadly publishes the software under a more permissive license this port should be considered licensed as such as well. Further, if you purchase a proprietary license for inclusion within commercial software under his Dual Licensing program your use of this software shall be under whatever terms he permits for that. Any contributions to this port must be made under equally permissive terms.

"Potrace" is a trademark of Peter Selinger.
