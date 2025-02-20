def get_x_pos(info):
    """
    From env returned info calculate x position of Mario.
    Parameters
    ----------
    info: dict of info returned by env.step()

    Returns x_pos: int
    -------

    """
    xscroll_hi = info.get("xscrollHi", 0)
    xscroll_lo = info.get("xscrollLo", 0)
    x_pos = (xscroll_hi << 8) + xscroll_lo
    return x_pos
