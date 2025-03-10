"""Static action sets for binary to discrete action space wrappers."""

# actions for very simple movement
SIMPLE_MOVEMENT = [
    ["NULL"],
    # ["A"],
    ["RIGHT"],
    ["RIGHT", "A"],
    ["LEFT"],
    # ["LEFT", "A"],
    ["B"],
    # ["A", "B"],
    ["RIGHT", "B"],
    # ["RIGHT", "A", "B"],
]

meaningful_actions = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No action
    # [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (Jump)
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Right
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Right + A (Jump)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Left
    # [0, 0, 0, 0, 0, 0, 1, 0, 1],  # Left + A (Jump)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Run)
    # [1, 0, 0, 0, 0, 0, 0, 0, 1],  # A + B (Jump + Run)
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Right + B (Run)
    # [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Right + A + B (Jump + Run)
]
