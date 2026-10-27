"""Static n=26 packing fixture for checking the evaluator installation.

It has the experiment's interface, ``main(timeout, current_best_solution)``.

These coordinates are supplied data, not the output of a search performed by
this module. This file is host-side documentation: it is never a seed or an
input to the isolated discovery experiment. A successful check establishes
that the evaluator can score this fixture, not that HELIX discovered it.
"""

import numpy as np

# Supplied coordinates from the earlier seed-230 solution with an ILS
# relocation of circle 11. Retained unchanged for evaluator checks.
KNOWN_BEST_CONFIG = [
    (0.4955317611, 0.7246573608, 0.1176296263),
    (0.8965327270, 0.5174044160, 0.1034671730),
    (0.9150736960, 0.9150736960, 0.0849262040),
    (0.7269056916, 0.4039573077, 0.1006003078),
    (0.9038486256, 0.3179199753, 0.0961512744),
    (0.1030605598, 0.5153991958, 0.1030604598),
    (0.2947460796, 0.6130764353, 0.1120770277),
    (0.4986680756, 0.4700365832, 0.1370103664),
    (0.2397105540, 0.2363264570, 0.0691806194),
    (0.8948174002, 0.7260471378, 0.1051824998),
    (0.0846395422, 0.9153604578, 0.0846394422),
    (0.1067901839, 0.7252166941, 0.1067900839),
    (0.2716298743, 0.4023652134, 0.0998982906),
    (0.8888437817, 0.1111562183, 0.1111561183),
    (0.7593523756, 0.2370411626, 0.0694401368),
    (0.3131158287, 0.0923915923, 0.0923914923),
    (0.5952197234, 0.2579505807, 0.0960189162),
    (0.4994283692, 0.0939273779, 0.0939272779),
    (0.4972844465, 0.9211395850, 0.0788603150),
    (0.2946095093, 0.8697788620, 0.1302210380),
    (0.0957323697, 0.3167414834, 0.0957322697),
    (0.7023095049, 0.8667413906, 0.1332585094),
    (0.7026095834, 0.6183341437, 0.1151488186),
    (0.4033587933, 0.2575829747, 0.0958422662),
    (0.1107790517, 0.1107790517, 0.1107789517),
    (0.6859430034, 0.0925921357, 0.0925920357),
]


def main(timeout, current_best_solution):
    """Return the supplied fixture; this function performs no search."""
    circles = np.array(KNOWN_BEST_CONFIG)
    return {"circles": circles, "all_scores": [float(circles[:, 2].sum())]}
