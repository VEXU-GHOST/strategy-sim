import json

with open("vid_inputs.json") as f:
    cfg = json.load(f)

RUNTIME  = cfg["video"]["runtime"]
STEP     = cfg["simulation"]["step_size"]
BOUNDARY = cfg["simulation"]["boundary"]
# etc.