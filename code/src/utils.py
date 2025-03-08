import os
import pickle


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


def load_trajectories(traj_folder):
    """
    Load human trajectories from a folder. Union
    all trajectories into a single list.

    Args:
        traj_folder: folder containing pickle file of trajectories

    Returns:
        trajectories: a list of trajectories, where each trajectory is a list of
            observations

    """
    states = list()
    actions = list()
    info = list()

    for file in os.listdir(traj_folder):
        if file.endswith(".pkl"):
            traj_path = os.path.join(traj_folder, file)

            with open(traj_path, "rb") as f:
                _t = pickle.load(f)
                states.extend([i["observation"] for i in _t])
                actions.extend([i["action"] for i in _t])
                info.extend([i["info"] for i in _t])

    assert len(states) == len(actions) == len(info)

    return states, actions, info
