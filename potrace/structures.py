class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point(%f, %f)" % (self.x, self.y)


class Segment:
    def __init__(self):
        self.tag = 0
        self.c = [Point(), Point(), Point()]
        self.vertex = Point()
        self.alpha = 0.0
        self.alpha0 = 0.0
        self.beta = 0.0


class Curve:
    def __init__(self, m):
        self.segments = [Segment() for i in range(m)]
        self.alphacurve = False

    def __len__(self):
        return len(self.segments)

    @property
    def n(self):
        return len(self)

    def __getitem__(self, item):
        return self.segments[item]


class Sums:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x2 = 0
        self.xy = 0
        self.y2 = 0


class Path:
    def __init__(self, pt, area, sign):
        self.pt = pt  # /* pt[len]: path as extracted from bitmap */

        self.area = area
        self.sign = sign
        self.next = None
        self.childlist = []
        self.sibling = []

        self._lon = []  # /* lon[len]: (i,lon[i]) = longest straight line from i */

        self._x0 = 0  # /* origin for sums */
        self._y0 = 0  # /* origin for sums */
        self._sums = []  # / *sums[len + 1]: cache for fast summing * /

        self._m = 0  # /* length of optimal polygon */
        self._po = []  # /* po[m]: optimal polygon */
        self._curve = []  # /* curve[m]: array of curve elements */
        self._ocurve = []  # /* ocurve[om]: array of curve elements */
        self._fcurve = []  # /* final curve: this points to either curve or ocurve.*/

    def __len__(self):
        return len(self.pt)

    def init_curve(self, m):
        pass
