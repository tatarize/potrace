from typing import Optional, Tuple, Union

import numpy

from .structures import *

POTRACE_TURNPOLICY_BLACK = 0
POTRACE_TURNPOLICY_WHITE = 1
POTRACE_TURNPOLICY_LEFT = 2
POTRACE_TURNPOLICY_RIGHT = 3
POTRACE_TURNPOLICY_MINORITY = 4
POTRACE_TURNPOLICY_MAJORITY = 5
POTRACE_TURNPOLICY_RANDOM = 6


# /* ---------------------------------------------------------------------- */

detrand_t = (
    # /* non-linear sequence: constant term of inverse in GF(8),
    #   mod x^8+x^4+x^3+x+1 */
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
)


def detrand(x: int, y: int) -> int:
    """deterministically and efficiently hash (x,y) into a pseudo-random bit"""
    # /* 0x04b3e375 and 0x05a8ef93 are chosen to contain every possible 5-bit sequence */
    z = ((0x04B3E375 * x) ^ y) * 0x05A8EF93
    z = (
        detrand_t[z & 0xFF]
        ^ detrand_t[(z >> 8) & 0xFF]
        ^ detrand_t[(z >> 16) & 0xFF]
        ^ detrand_t[(z >> 24) & 0xFF]
    )
    return z


def majority(bm: numpy.array, x: int, y: int) -> int:
    """
     /* return the "majority" value of bitmap bm at intersection (x,y). We
    assume that the bitmap is balanced at "radius" 1.  */
    """
    for i in range(2, 5):  # /* check at "radius" i */
        ct = 0
        for a in range(-i + 1, i - 2):
            try:
                ct += 1 if bm[y + i - 1][x + a] else -1
            except IndexError:
                pass
            try:
                ct += 1 if bm[y + a - 1][x + i - 1] else -1
            except IndexError:
                pass
            try:
                ct += 1 if bm[y - i][x + a - 1] else -1
            except IndexError:
                pass
            try:
                ct += 1 if bm[y + a][x - i] else -1
            except IndexError:
                pass
        if ct > 0:
            return 1
        elif ct < 0:
            return 0
    return 0


"""
/* ---------------------------------------------------------------------- */
/* decompose image into paths */
"""


def xor_to_ref(bm: numpy.array, x: int, y: int, xa: int) -> None:
    """
     /* efficiently invert bits [x,infty) and [xa,infty) in line y. Here xa
    must be a multiple of BM_WORDBITS. */
    """

    if x < xa:
        bm[y, x:xa] ^= True
    elif x != xa:
        bm[y, xa:x] ^= True


def xor_path(bm: numpy.array, p: Path) -> None:
    """
    a path is represented as an array of points, which are thought to
    lie on the corners of pixels (not on their centers). The path point
    (x,y) is the lower left corner of the pixel (x,y). Paths are
    represented by the len/pt components of a path_t object (which
    also stores other information about the path) */

    xor the given pixmap with the interior of the given path. Note: the
    path must be within the dimensions of the pixmap.
    """
    if len(p) <= 0:  # /* a path of length 0 is silly, but legal */
        return

    y1 = p.pt[-1].y
    xa = p.pt[0].x
    for n in p.pt:
        x, y = n.x, n.y
        if y != y1:
            # /* efficiently invert the rectangle [x,xa] x [y,y1] */
            xor_to_ref(bm, x, min(y, y1), xa)
            y1 = y


def findpath(bm: numpy.array, x0: int, y0: int, sign: bool, turnpolicy: int) -> Path:
    """
    /* compute a path in the given pixmap, separating black from white.
    Start path at the point (x0,x1), which must be an upper left corner
    of the path. Also compute the area enclosed by the path. Return a
    new path_t object, or NULL on error (note that a legitimate path
    cannot have length 0). Sign is required for correct interpretation
    of turnpolicies. */"""

    x = x0
    y = y0
    dirx = 0
    diry = -1  # diry-1
    pt = []
    area = 0

    while True:  # /* while this path */
        # /* add point to path */
        pt.append(Point(int(x), int(y)))

        # /* move to next point */
        x += dirx
        y += diry
        area += x * diry

        # /* path complete? */
        if x == x0 and y == y0:
            break

        # /* determine next direction */
        cy = y + (diry - dirx - 1) // 2
        cx = x + (dirx + diry - 1) // 2
        try:
            c = bm[cy][cx]
        except IndexError:
            c = 0
        dy = y + (diry + dirx - 1) // 2
        dx = x + (dirx - diry - 1) // 2
        try:
            d = bm[dy][dx]
        except IndexError:
            d = 0

        if c and not d:  # /* ambiguous turn */
            if (
                turnpolicy == POTRACE_TURNPOLICY_RIGHT
                or (turnpolicy == POTRACE_TURNPOLICY_BLACK and sign)
                or (turnpolicy == POTRACE_TURNPOLICY_WHITE and not sign)
                or (turnpolicy == POTRACE_TURNPOLICY_RANDOM and detrand(x, y))
                or (turnpolicy == POTRACE_TURNPOLICY_MAJORITY and majority(bm, x, y))
                or (
                    turnpolicy == POTRACE_TURNPOLICY_MINORITY and not majority(bm, x, y)
                )
            ):
                tmp = dirx  # /* right turn */
                dirx = diry
                diry = -tmp
            else:
                tmp = dirx  # /* left turn */
                dirx = -diry
                diry = tmp
        elif c:  # /* right turn */
            tmp = dirx
            dirx = diry
            diry = -tmp
        elif not d:  # /* left turn */
            tmp = dirx
            dirx = -diry
            diry = tmp

    # /* allocate new path object */
    return Path(pt, area, sign)


def findnext(bm: numpy.array) -> Optional[Tuple[Union[int], int]]:
    """
    /* find the next set pixel in a row <= y. Pixels are searched first
       left-to-right, then top-down. In other words, (x,y)<(x',y') if y>y'
       or y=y' and x<x'. If found, return 0 and store pixel in
       (*xp,*yp). Else return 1. Note that this function assumes that
       excess bytes have been cleared with bm_clearexcess. */
    """
    w = numpy.nonzero(bm)
    if len(w[0]) == 0:
        return None

    q = numpy.where(w[0] == w[0][-1])
    y = w[0][q]
    x = w[1][q]
    return y[0], x[0]


def setbbox_path(p: Path):
    """
     /* Find the bounding box of a given path. Path is assumed to be of
    non-zero length. */
    """
    y0 = float("inf")
    y1 = 0
    x0 = float("inf")
    x1 = 0
    for k in range(len(p)):
        x = p.pt[k].x
        y = p.pt[k].y

        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    return x0, y0, x1, y1


def pathlist_to_tree(plist: list, bm: numpy.array) -> None:
    """
    /* Give a tree structure to the given path list, based on "insideness"
       testing. I.e., path A is considered "below" path B if it is inside
       path B. The input pathlist is assumed to be ordered so that "outer"
       paths occur before "inner" paths. The tree structure is stored in
       the "childlist" and "sibling" components of the path_t
       structure. The linked list structure is also changed so that
       negative path components are listed immediately after their
       positive parent.  Note: some backends may ignore the tree
       structure, others may use it e.g. to group path components. We
       assume that in the input, point 0 of each path is an "upper left"
       corner of the path, as returned by bm_to_pathlist. This makes it
       easy to find an "interior" point. The bm argument should be a
       bitmap of the correct size (large enough to hold all the paths),
       and will be used as scratch space. Return 0 on success or -1 on
       error with errno set. */
    """
    # path_t *p, *p1
    # path_t *heap, *heap1
    # path_t *cur
    # path_t *head
    # path_t **plist_hook          #/* for fast appending to linked list */
    # path_t **hook_in, **hook_out #/* for fast appending to linked list */
    # bbox_t bbox

    bm = bm.copy()

    # /* save original "next" pointers */

    for p in plist:
        p.sibling = p.next
        p.childlist = None

    heap = plist
    """
      /* the heap holds a list of lists of paths. Use "childlist" field
         for outer list, "next" field for inner list. Each of the sublists
         is to be turned into a tree. This code is messy, but it is
         actually fast. Each path is rendered exactly once. We use the
         heap to get a tail recursive algorithm: the heap holds a list of
         pathlists which still need to be transformed. */
    """
    while heap:
        # /* unlink first sublist */
        cur = heap
        heap = heap.childlist
        cur.childlist = None

        # /* unlink first path */
        head = cur
        cur = cur.next
        head.next = None

        # /* render path */
        xor_path(bm, head)
        x0, y0, x1, y1 = setbbox_path(head)

        """
        /* now do insideness test for each element of cur; append it to
       head->childlist if it's inside head, else append it to
       head->next. */
       """
        # hook_in= head.childlist
        # hook_out= head.next
        # list_forall_unlink(p, cur):
        #     if p.priv.pt[0].y <= bbox.y0:
        #         list_insert_beforehook(p, hook_out)
        #         # /* append the remainder of the list to hook_out */
        #         hook_out = cur
        #         break
        #     if (BM_GET(bm, p.priv.pt[0].x, p.priv.pt[0].y-1)):
        #         list_insert_beforehook(p, hook_in)
        #     else:
        #         list_insert_beforehook(p, hook_out)

    # /* clear bm */
    # clear_bm_with_bbox(bm, &bbox)
    #
    # #/* now schedule head->childlist and head->next for further processing */
    # if head.next:
    #     head.next.childlist = heap
    #     heap = head.next
    # if head.childlist:
    #     head.childlist.childlist = heap
    #     heap = head->childlist
    #
    # #/* copy sibling structure from "next" to "sibling" component */
    # p = plist
    # while p:
    #     p1 = p.sibling
    #     p.sibling = p.next
    #     p = p1
    #
    # """
    # /* reconstruct a new linked list ("next") structure from tree
    # ("childlist", "sibling") structure. This code is slightly messy,
    # because we use a heap to make it tail recursive: the heap
    # contains a list of childlists which still need to be
    # processed. */"""
    # heap = plist
    # if heap:
    #     heap.next = None  #/* heap is a linked list of childlists */
    # plist = None
    # plist_hook = plist
    # while heap:
    #     heap1 = heap.next
    #     for (p=heap; p; p=p->sibling):
    #         #/* p is a positive path */
    #         #/* append to linked list */
    #         list_insert_beforehook(p, plist_hook)
    #
    #         #/* go through its children */
    #     for (p1=p->childlist; p1; p1=p1->sibling):
    #         #/* append to linked list */


#         list_insert_beforehook(p1, plist_hook);
#         #/* append its childlist to heap, if non-empty */
#     if (p1.childlist):
#         list_append(path_t, heap1, p1->childlist)
#     heap = heap1


def bm_to_pathlist(
    bm: numpy.array, turdsize: int = 2, turnpolicy: int = POTRACE_TURNPOLICY_MINORITY
) -> list:
    """
    /* Decompose the given bitmap into paths. Returns a linked list of
    path_t objects with the fields len, pt, area, sign filled
    in. Returns 0 on success with plistp set, or -1 on error with errno
    set. */
    """
    plist = []  # /* linked list of path objects */
    original = bm.copy()

    """/* be sure the byte padding on the right is set to 0, as the fast
    pixel search below relies on it */"""
    # /* iterate through components */
    while True:
        n = findnext(bm)
        if n is None:
            break
        y, x = n
        # /* calculate the sign by looking at the original */
        sign = original[y][x]
        # /* calculate the path */
        path = findpath(bm, x, y + 1, sign, turnpolicy)
        if path is None:
            raise ValueError

        # /* update buffered image */
        xor_path(bm, path)

        # /* if it's a turd, eliminate it, else append it to the list */
        if path.area > turdsize:
            plist.append(path)

    # pathlist_to_tree(plist, original)
    return plist
