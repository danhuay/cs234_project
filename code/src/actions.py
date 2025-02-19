"""Static action sets for binary to discrete action space wrappers."""

# actions for the simple run RIGHT environment
RIGHT_ONLY = [["NOOP"], ["RIGHT"], ["RIGHT", "A"], ["RIGHT", "B"], ["RIGHT", "A", "B"]]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["RIGHT"],
    ["RIGHT", "A"],
    ["RIGHT", "B"],
    ["RIGHT", "A", "B"],
    ["A"],
    ["LEFT"],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ["NOOP"],
    ["RIGHT"],
    ["RIGHT", "A"],
    ["RIGHT", "B"],
    ["RIGHT", "A", "B"],
    ["A"],
    ["LEFT"],
    ["LEFT", "A"],
    ["LEFT", "B"],
    ["LEFT", "A", "B"],
    ["DOWN"],
    ["UP"],
]
