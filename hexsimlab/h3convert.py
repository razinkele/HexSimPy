raise NotImplementedError(
    "hexsimlab.h3convert is an incomplete prototype. "
    "The 'hex_to_latlon' function is undefined."
)

import h3


def convert_to_h3(grid, cell_size, origin, resolution):

    h3cells = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            lat, lon = hex_to_latlon(r, c, cell_size, origin)

            h3cells.append(h3.geo_to_h3(lat, lon, resolution))

    return h3cells
