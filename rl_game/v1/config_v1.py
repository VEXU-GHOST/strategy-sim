# ============================================================
# V1 strategy settings only.
#
# IMPORTANT:
# Do not duplicate the real field layout here. Robot positions,
# objects, goals, toggles, and scoring state should come from
# the teammate's full GameEnv through GameEnvBridge.
# ============================================================

EPISODE_TIME = 60.0
ACTIVE_AGENTS = ("A0", "A1")

# High-level V1 regions. The current bridge maps them to the
# four geometric quadrants of the full simulator board.
REGIONS = ("q0", "q1", "q2", "q3")

# Temporary timing model for macro actions.
# Later, replace TimedSimExecutor with an executor that drives
# the full low-level simulator or the real robot skills.
DRIVE_SPEED_CELLS_PER_SECOND = 20.0
COLLECT_TIME = 1.0
SCORE_TIME = 1.5
TOGGLE_TIME = 1.0
WAIT_TIME = 1.0

# The current develop-branch reward code treats Team A as the
# blue score (index 1 from calcGoalScore()). Keep this isolated
# here so it is easy to change if the team standardizes mapping.
TEAM_A_SCORE_INDEX = 1
