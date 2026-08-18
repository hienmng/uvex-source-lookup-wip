import warnings
warnings.filterwarnings('ignore', 'Wswiglal-redir-stdio')

from astropy import units as u
from astropy.coordinates import search_around_sky, SkyCoord, SkyOffsetFrame
from astropy.table import QTable
from astropy_healpix import HEALPix
from functools import partial
from ligo.skymap import plot # v 2.5.4, `_add_newdoc_ufunc` removed from numpy v >=2.5
from ligo.skymap.plot.poly import cut_prime_meridian, subdivide_vertices
from matplotlib import pyplot as plt

import healpy as hp
import numpy as np

# TODO: def roll_at(center, time): # unncessary if roll is passed parallel to center
# TODO: def pad(): # padding variance per source? 

def offset_coords(center, coords, roll=0*u.deg):
    """Offset-frame (lon, lat) of `coords` about `center`, rotated by `roll`.

    Analytic and vectorized equivalent of coords.transform_to(SkyOffsetFrame(origin=center,
    rotation=roll)).
    """
    # Polar coordinates (rho, pa) about the center in the offset frame
    rho = center.separation(coords)
    pa = center.position_angle(coords) - roll # Bearing of source as seen from fov center,
                                              # measured East of North

    # Convert (rho, pa) to the SkyOffsetFrame's (lon, lat),
    # equivalent of s.transform_to(SkyOffsetFrame(origin=c))
    lat = np.arcsin(np.sin(rho) * np.cos(pa)).to(u.deg)
    lon = np.arctan2(np.sin(rho) * np.sin(pa), np.cos(rho)).to(u.deg) # Returns [-pi, pi],
                                                                      # no wrapping needed
    return lon, lat

def _inside_box(lon, lat, half_width):
    return (np.abs(lon) <= half_width) & (np.abs(lat) <= half_width)
    
def fields_with_sources(center_fovs, source_coords, fov=3.4*u.deg, roll=0*u.deg, pad=0*u.deg):
    """Write full description later

    Parameters
    ----------
    center_fovs: astropy.coordinates.SkyCoord
        Center of the field of view
    source_coords: astropy.coordinates.SkyCoord
        Center of the sources
    roll: astropy.units.Quantity
        Position angle of the spacecraft (default 0 degrees)
    fov: astropy.units.Quantity
        Full edge length of the field of view footprint (default 3.4 degrees)
    pad: astropy.units.Quantity
        Padding for angular distance (default 0 degrees).
    
    Returns
    -------
    i_src_in: ndarray
        Positional indices into `source_coords`. Not source IDs.
    i_fld_in: ndarray
        Positional indices into `center_fovs`. Not field IDs.
        Parallel to i_src_in: pair k is (source_coords[i_src_in[k]]),
        center_fovs[i_fov_in[k]]). A source appears once per containing
        field, so indices repeat where the grid overlaps.
    """

    # Geometric constants
    half_width = fov / 2 + pad              # Half-width, i.e. distance from fov center to each fov edge
    half_diagonal = half_width * np.sqrt(2) # Half-diagonal, i.e. distance from fov center to each fov corner
                                            # Radius of smallest circle enclosing the rectangle (below)
                                            # Roll-invariant

    # --- Cone search
    # Positional indices into fov center array and source center array, respectively.
    i_fov, i_src, _, _ = search_around_sky(coords1=center_fovs, coords2=source_coords,
                                           seplimit=half_diagonal)
    # NOTE: module-level FUNCTION (current) takes in (coords1, coords2) and returns (idx1, idx2),
    # whereas METHOD a.search_around_sky() returns indices in the opposite order.

    # Raise error if indices are assigned to incorrect arrays
    assert np.all(center_fovs[i_fov].separation(source_coords[i_src]) <= half_diagonal)

    # --- Rectangular test
    c = center_fovs[i_fov]
    s = source_coords[i_src]

    # Convert to offset frame
    lon, lat = offset_coords(center=c, coords=s, roll=roll)

    # Box containment test in the offset frame
    mask = _inside_box(lon=lon, lat=lat, half_width=half_width)

    i_src_in, i_fov_in = i_src[mask], i_fov[mask]

    return i_src_in, i_fov_in

def sources_in_field(center, source_coords, **kw):
    i_src, _ = fields_with_sources(center.reshape(1), source_coords, **kw)
    return i_src

def build_result(i_src, i_fld, source_table, field_table,
                 source_cols=None, field_cols=None):
    """Join matched indices into a flat QTable, one row per (source, field)."""
    return QTable({
        (f'{source_col}': source_table[source_col][i_src] for source_col in source_cols),
        (f'{field_col}': field_table[field_col][i_fld] for field_col in field_cols)
    })

def get_footprint_polygon(center, fov=3.4*u.deg, roll=0*u.deg):
    """Get the footprint of the field of view for a given orientation.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        The center of the field of view.
    roll : astropy.units.Quantity
        The position angle (optional, default 0 degrees).
    
    Returns
    -------
    astropy.coordinates.SkyCoord
        A sky coordinate array of shape (..., 4) giving the four verticies of the footprint.
    """

    frame = SkyOffsetFrame(origin=center, rotation=roll)

    # Four corner offsets in the local frame
    # TL > BL > BR > TR (for increasing East)
    lon = np.asarray([0.5,  0.5, -0.5, -0.5]) * fov
    lat = np.asarray([0.5, -0.5, -0.5,  0.5]) * fov
    skycoord = SkyCoord(
        np.tile(lon[(None,) * frame.ndim], (*frame.shape, 1)),
        np.tile(lat[(None,) * frame.ndim], (*frame.shape, 1)),
        frame=frame[..., None]
    ).icrs

    assert skycoord.shape == (*np.broadcast_shapes(frame.shape, np.shape(roll)), 4)

    return skycoord

def get_footprint_healpix(center, rotate=None):
    """Get the HEALPix footprint of the field of view for a given orientation.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        The center of the field of view.
    rotate : astropy.units.Quantity
        The position angle (optional, default 0 degrees).

    Returns
    -------
    np.ndarray
        An array of HEALPix indices contained within the footprint.
    """
    xyz = get_footprint_polygon(center, rotate=rotate).cartesian.xyz.value
    idx_healpix = hp.query_polygon(healpix.nside, xyz.T,
                                   nest=(healpix.order == 'nested'))
    """
    hp.query_polygon() normalizes winding internally,
    but MOC libraries and spherical_geometry do not.
    Re-check orientation of output if passing vertices.
    """
    return idx_healpix

def get_footprint_grid(centers, rolls=None):
    """Calculate the HEALPix footprints of all pointings on the grid.

    Returns
    -------
    generator
        A generator that yields the indices for each pointing center and for
        each roll.
    """
    if rolls is None:
        rolls = [0] * u.deg
    poly = get_footprint_polygon(centers[:, None], rotate=rolls[None, :])
    xyz = np.moveaxis(poly.cartesian.xyz.value, 0, -1) # (N, M, 4, 3)
    query_polygon = partial(
        hp.query_polygon, healpix.nside, nest=(healpix.order == 'nested'))
    return ((query_polygon(_) for _ in __) for __ in xyz)

def plot_sources_in_field(center, colors, rolls, radius, subdiv=50,
                          sources=None, field_sources=None):
    fig = plt.figure(figsize=(6, 6), layout='constrained')
    ax = fig.add_subplot(projection='astro zoom', center=center, radius=radius)
    ax.grid()

    if sources is not None:
        ax.plot(sources.ra.deg, sources.dec.deg, '.', ms=1, color='0.75',
                transform=ax.get_transform('world'), zorder=1, label='nearby')

    if field_sources is not None:
        ax.plot(field_sources.ra.deg, field_sources.dec.deg, '.', ms=2,
                color='tab:red', transform=ax.get_transform('world'),
                zorder=2, label=f'in field (N={len(field_sources):,})')

    for roll, color in zip(rolls, colors):
        poly = get_footprint_polygon(center, rotate=roll)
        verts = np.column_stack((poly.ra.deg, poly.dec.deg))
        if subdiv:
            verts = subdivide_vertices(verts, subdiv)
        ax.add_patch(plt.Polygon(verts, transform=ax.get_transform('world'),
                                 facecolor='none', edgecolor=color, lw=1.5,
                                 zorder=3,
                                 label=f'{roll.value:.0f}$^{{\\circ}}$'))
        ax.plot(poly.ra.deg[0], poly.dec.deg[0], 'o', color=color, ms=4,
                transform=ax.get_transform('world'), zorder=4)
    
    ax.plot(center.ra.deg, center.dec.deg, 'x', color='black', ms=8,
            transform=ax.get_transform('world'), zorder=4)
    
    ax.legend(loc='upper right', markerscale=4)
    
    return fig, ax