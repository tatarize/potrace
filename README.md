# Python Potrace
Pure Python Port of Potrace. This is a python port of Peter Selinger's Potrace (based on 1.16 code).

<img width="200" height="200" src="https://gist.githubusercontent.com/tatarize/42884e5e99cda88aa5ddc2b0ab280973/raw/488cafa1811bd2227458390804910fbc4a90b9ea/head.svg"/>

![head-orig3](https://user-images.githubusercontent.com/3302478/115929160-2757f180-a43c-11eb-88dc-1320706c9a3f.png)
> Note: This image has been scaled up by a factor of 3.

![head-scaled](https://user-images.githubusercontent.com/3302478/154810339-6a444bfa-3f2e-4ad0-91cf-40570838a918.png)

This port is needed because many python hooks to the original code such as `pypotrace` have installation issues and compilation problems with some OSes. This potrace is written in pure python and will be compatible with basically anything.

# Installing

To install or use as a requirement:
* `pip install potracer`

### Potrace-CLI
If you wish to use the Command Line Interface that is stored in a sister project `potrace-cli` (https://github.com/tatarize/potrace-cli). This can be installed with:
* `pip install potracer[cli]`

or:

* `pip install potrace-cli`

The cli project contains valid console script entrypoints for potrace. If you install the command-line package it will add `potracer` to your console scripts. Note the `-r` suffix so that it does not interfere with potrace that may be otherwise installed.

# Requirements
* numpy: for bitmap structures.

# Speed
Being written in python this code may be about 500x slower than the pure-c potrace. It is however fast enough for general use.

# How To Use

This is a barebones script that uses `potracer` to convert a given file to an svg. The file loading is performed with `Pillow`. The file saving is done iterating over the curves inside the plist to produce the SVG with the corners and curves of the potrace structures.

Note: The `bm.trace()` is the primary function call needed for potrace.
```python
import sys
from PIL import Image
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY  # `potracer` library


def file_to_svg(filename: str):
    try:

        image = Image.open(filename)
    except IOError:
        print("Image (%s) could not be loaded." % filename)
        return
    bm = Bitmap(image, blacklevel=0.5)
    # bm.invert()
    plist = bm.trace(
        turdsize=2,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1,
        opticurve=False,
        opttolerance=0.2,
    )
    with open(f"{filename}.svg", "w") as fp:
        fp.write(
            f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
        parts = []
        for curve in plist:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
        fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
        fp.write("</svg>")


if __name__ == '__main__':
    file_to_svg(sys.argv[1])
```

# Parallel Projects
This project intentionally duplicates a considerable amount of the API of `pypotrace` such that this library can be used as a drop-in replacement.

To permit performing potrace commands from the commandline, this library offers CLI potrace as an optional package.


# License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2, or (at your option) any later version.

Furthermore, this is permitted to be relicensed under any terms the Peter Selinger's original Potrace is licensed under. If he broadly publishes the software under a more permissive license this port should be considered licensed as such as well. Further, if you purchase a proprietary license for inclusion within commercial software under his Dual Licensing program your use of this software shall be under whatever terms he permits for that. Any contributions to this port must be made under equally permissive terms.

"Potrace" is a trademark of Peter Selinger. Permission granted by Peter Selinger.
