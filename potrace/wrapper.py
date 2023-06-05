from PIL import Image

from .potrace import Bitmap
from .potrace import POTRACE_CORNER, Path
from .backend_svg import backend_svg, backend_jagged_svg
from .backend_svg import register as svg_register

def get_default_settings():
    settings = {
        'output': "out.svg", # write all output to this file
        'backend': "svg", # select backend by name
        'turnpolicy': "minority", # how to resolve ambiguities in path decomposition
        'turdsize': 2, # suppress speckles of up to this size
        'alphamax': 1.0, # corner threshold parameter
        'longcurve': False, # turn off curve optimization
        'opttolerance': 0.2, # curve optimization tolerance
        'invert': False, # invert bitmap
        'color': "#000000", # set foreground color (default Black)
        'blacklevel': 0.5 # black level
    }
    return settings

def convert_png_to_svg(input_filename, **settings):
    output = settings.get('output', "out.svg") 
    backend = settings.get('backend', "svg") 
    turnpolicy = settings.get('turnpolicy', "minority") 
    turdsize = settings.get('turdsize', 2) 
    alphamax = settings.get('alphamax', 1.0) 
    longcurve = settings.get('longcurve', False) 
    opttolerance = settings.get('opttolerance', 0.2) 
    invert = settings.get('invert', False) 
    color = settings.get('color', "#000000") 
    blacklevel = settings.get('blacklevel', 0.5) 

    try:
        image = Image.open(input_filename)
    except IOError:
        print("Image (%s) could not be loaded." % input_filename)
        return
    bm = Bitmap(image, blacklevel=blacklevel)
    if invert:
        bm.invert()
    plist = bm.trace(
        turdsize=turdsize,
        turnpolicy=turnpolicy,
        alphamax=alphamax,
        opticurve=not longcurve,
        opttolerance=opttolerance,
    )
    if output:
        backends = {}
        svg_register(backends)
        
        if backend:
            output = backends[backend]
            output(settings, image, plist)
        else:
            print("No backends exist to process output.")
