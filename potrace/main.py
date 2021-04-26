"""
Copyright (C) 2001-2019 Peter Selinger.
This file is part of Potrace. It is free software and it is covered
by the GNU General Public License. See the file COPYING for details.

Python Port by Tatarize, April 2021
"""

import argparse
import sys
import pkg_resources

from PIL import Image
import numpy as np

from .decompose import bm_to_pathlist
from .tracer import process_path


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

plugin_register_functions = []
for entry_point in pkg_resources.iter_entry_points("potrace.backends"):
    try:
        plugin = entry_point.load()
        plugin_register_functions.append(plugin)
    except pkg_resources.DistributionNotFound:
        pass


if len(plugin_register_functions) == 0:
    """
    Fallback if entry points are not permitted.
    """
    from .backend_svg import register as svg_register
    plugin_register_functions.append(svg_register)

backends = {}
for register in plugin_register_functions:
    register(backends)

if len(backends) != 0:
    choices = [b for b in backends]
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=choices,
        default='svg' if 'svg' in choices else choices[0],
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
    "-O", "--opttolerance", type=float, help="curve optimization tolerance", default=0.2
)
parser.add_argument(
    "-C",
    "--color",
    type=str,
    help="set foreground color (default Black)",
    default="#000000",
)
parser.add_argument(
    "-i",
    "--invert",
    action="store_true",
    help="invert bitmap",
)
parser.add_argument(
    "-k",
    "--blacklevel",
    type=float,
    default=0.5,
    help="invert bitmap",
)
parser.add_argument(
    "-s",
    "--scale",
    type=int,
    default=1,
    help="Scale the image by an integer factor n>0.",
)
parser.add_argument(
    "-1",
    "--dither",
    action="store_true",
    help="Dither rather than threshold to 1-bit.",
)

def timeit(method):
    def timed(*args, **kw):
        import time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

@timeit
def run():
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    turnpolicy = choices.index(args.turnpolicy)
    if args.version:
        print("Python Potrace 0.0.1")
        return
    if args.license:
        try:
            with open("LICENSE", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.endswith('\n'):
                        print(line[0:-1])
                    else:
                        print(line)
        except IOError:
            print("License not found.")
        return
    if args.output:
        try:
            output = backends[args.backend]
        except AttributeError:
            print("No backends exist to process output.")
            return
    if args.filename:
        try:
            image = Image.open(args.filename)
        except IOError:
            print("Image (%s) could not be loaded." % args.filename)
            return
        if args.scale != 1:
            scale = args.scale
            if isinstance(image, Image.Image):
                image = image.resize((scale * image.width, scale * image.height), Image.BICUBIC)
        if args.dither:
            image = image.convert("1")
        else:
            if image.mode != "L":
                image = image.convert("L")
            if args.invert:
                points = lambda e: 255 if (e/255.0) < args.blacklevel else 0
            else:
                points = lambda e: 0 if (e / 255.0) < args.blacklevel else 255
            image = image.point(points)
            image = image.convert("1")
            bm = np.invert(image)
            bm = np.pad(bm, [(0, 1), (0, 1)], mode='constant')

        plist = bm_to_pathlist(bm, turdsize=args.turdsize, turnpolicy=turnpolicy)
        process_path(
            plist,
            alphamax=args.alphamax,
            opticurve=not args.longcurve,
            opttolerance=args.opttolerance,
        )
        if args.output:
            output(args, image, plist)
    else:
        print("No image loaded.\n 'potrace --help' for help.")
