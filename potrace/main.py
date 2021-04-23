"""
/* Copyright (C) 2001-2019 Peter Selinger.
     This file is part of Potrace. It is free software and it is covered
     by the GNU General Public License. See the file COPYING for details. */

/* transform jaggy paths into smooth curves */
"""
import argparse
import sys

from PIL import Image

from .decompose import bm_to_pathlist
from .tracer import process_path, POTRACE_CORNER

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v", "--version", action="store_true", help="prints version info and exit"
)
parser.add_argument(
    "-l", "--license", action="store_true", help="prints license info and exit"
)
parser.add_argument("filename", nargs="?", type=str, help="an input file")
parser.add_argument(
    "-o", "--output", type=str, help="write all output to this file", default="out.svg"
)
choices = ["svg"]
parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=choices,
    default="svg",
    help="select backend by name",
)
choices = ["black", "white", "left", "right", "minority", "majority", "random"]
parser.add_argument(
    "-z",
    "--turnpolicy",
    type=str,
    choices=choices,
    default="minority",
    help="how to resolve ambiguities in path decomposition",
)
parser.add_argument(
    "-t",
    "--turdsize",
    type=int,
    help="suppress speckles of up to this size (default 2)",
    default=2,
)
parser.add_argument(
    "-a", "--alphamax", type=float, help="corner threshold parameter", default=1
)
parser.add_argument(
    "-n", "--longcurve", action="store_true", help="turn off curve optimization"
)
parser.add_argument(
    "-D", "--decompose_only", action="store_true", help="only decompose the image"
)
parser.add_argument(
    "-O", "--opttolerance", type=float, help="curve optimization tolerance", default=0.2
)
parser.add_argument(
    "-u", "--unit", type=int, help="quantize output to 1/unit pixels", default=10
)

parser.add_argument(
    "-C",
    "--color",
    type=str,
    help="set foreground color (default Black)",
    default="#000000",
)
parser.add_argument(
    "-g", "--group", action="store_true", help="group related paths together"
)
parser.add_argument(
    "-f", "--flat", action="store_true", help="whole image as a single path"
)


def run():
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    turnpolicy = choices.index(args.turnpolicy)
    if args.version:
        print("Python Potrace 0.0.1")
        return
    if args.filename:
        image = Image.open(args.filename)
        if image.mode != "L":
            image = image.convert("L")
        image = image.point(lambda e: int(e > 127) * 255)
        image = image.convert("1")

        plist = bm_to_pathlist(image, turdsize=args.turdsize, turnpolicy=turnpolicy)
        process_path(
            plist,
            alphamax=args.alphamax,
            opticurve=not args.longcurve,
            opttolerance=args.opttolerance,
        )
        if args.output:
            with open(args.output, "w") as fp:
                fp.write(
                    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 %d %d">'
                    % (image.width, image.height)
                )
                parts = []
                if args.decompose_only:
                    parts.append("M")
                    for path in plist:
                        for point in path.pt:
                            parts.append(" %f,%f" % (point.x, point.y))
                else:
                    for path in plist:
                        fs = path._fcurve[-1].c[2]
                        parts.append("M%f,%f" % (fs.x, fs.y))
                        for segment in path._fcurve.segments:
                            if segment.tag == POTRACE_CORNER:
                                a = segment.c[1]
                                parts.append("L%f,%f" % (a.x, a.y))
                                b = segment.c[2]
                                parts.append("L%f,%f" % (b.x, b.y))
                            else:
                                a = segment.c[0]
                                b = segment.c[1]
                                c = segment.c[2]
                                parts.append("C%f,%f %f,%f %f,%f" % (a.x, a.y, b.x, b.y, c.x, c.y))
                        parts.append('z')
                fp.write('<path stroke="none" fill="black" fill-rule="evenodd" d="%s"/>' % "".join(parts))
                fp.write("</svg>")
    else:
        print("No image loaded.")

# M346.430 287.028
# L 410.012 221.556 402.662 214.028
# C 398.620 209.888,370.429 180.850,340.015 149.500
# C 309.602 118.150,281.744 89.470,278.109 85.767
# L 271.500 79.034 202.774 149.767
# C 164.975 188.670,133.980 220.893,133.897 221.373
# C 133.815 221.853,164.738 254.139,202.617 293.120
# L 271.486 363.994 277.167 358.247
# C 280.291 355.086,311.460 323.038,346.430 287.028
run()
