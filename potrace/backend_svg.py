
"""
Plugins must have a register function that accepts **kwargs, (for forwards compatibility).

The value backends will be the dict of backends used by potrace. A plugin *should* register
itself as a valid backend.

These are registered in the pip: setup.cfg

[options.entry_points]
potrace.backends = SVG = potrace.backend_svg:register
"""
from potrace.tracer import POTRACE_CORNER


def register(backends: dict, **kwargs):
    backends["svg"] = backend_svg
    backends["jagged-svg"] = backend_jagged_svg


def backend_svg(args, image, plist):
    with open(args.output, "w") as fp:
        fp.write(
            '<svg version="1.1"'
            ' xmlns="http://www.w3.org/2000/svg"'
            ' xmlns:xlink="http://www.w3.org/1999/xlink"'
            ' viewBox="0 0 %d %d">'
            % (image.width, image.height)
        )
        parts = []
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
                    parts.append(
                        "C%f,%f %f,%f %f,%f"
                        % (a.x, a.y, b.x, b.y, c.x, c.y)
                    )
            parts.append("z")
        fp.write(
            '<path stroke="none" fill="%s" fill-rule="evenodd" d="%s"/>'
            % (args.color, "".join(parts))
        )
        fp.write("</svg>")


def backend_jagged_svg(args, image, plist):
    with open(args.output, "w") as fp:
        fp.write(
            '<svg version="1.1"'
            ' xmlns="http://www.w3.org/2000/svg"'
            ' xmlns:xlink="http://www.w3.org/1999/xlink"'
            ' viewBox="0 0 %d %d">'
            % (image.width, image.height)
        )
        parts = []
        parts.append("M")
        for path in plist:
            for point in path.pt:
                parts.append(" %f,%f" % (point.x, point.y))
            parts.append("z")
        fp.write(
            '<path stroke="none" fill="%s" fill-rule="evenodd" d="%s"/>'
            % (args.color, "".join(parts))
        )
        fp.write("</svg>")
