# Important: game fidelity vs strategy prototype

This strategy simulator intentionally does **not** claim to be an exact physics or referee simulator. It is designed to let the team test high-level decisions while the complete `GameEnv` is being developed.

## Aligned with published 2026–27 VEX U Override v2.1

- Four robots total, two per alliance.
- 9 goals, 4 toggles.
- 30 s Autonomous + 90 s Driver (120 s total). Endgame is the last 10 s.
- Alliance-colored *visible pin half* = 5 match points, owned yellow *visible pin half* = 10 points, each robot finishing midfield = 8 points.
- Autonomous winner gets 12 match points (6 each on a tie); AWP is a separate tournament Win Point, never included in match score.
- VEX U autonomous AWP thresholds: 12 scored halves, 4 goals with >=2 scored halves, >=1 robot in midfield, neither robot touching perimeter (subject to official rule specifics and violation conditions).
- Yellow Pin ownership depends on Toggle in its Quadrant, or majority robots in Midfield for the central Goal.
- Cups facilitate pin stacking and do not earn direct match points.

## Deliberate simplifications to revisit after simulator integration

1. **Goal stacking:** One visible Pin half per scored physical Pin; Pin→Cup→Pin sequence, simplified max stack capacity. Official game can involve multiple visible halves, orientations, nested overlaps, physical occlusion, geometry and referee judgement.
2. **Object types:** `alliance_pin`, `yellow_pin`, `cup`. The official field has red/yellow, blue/yellow, red/blue, and yellow/yellow Pins; red/blue types, orientation and fine visibility are abstracted.
3. **Field:** 12x12 coordinates; the 9 goal positions, 6 abstract source piles and 4 toggles are **illustrative**, not an authoritative field CAD or exact loader setup. The four neutral short goals begin with a yellow Pin surrogate.
4. **Movement:** Euclidean distance/speed + configured overhead; no collision, obstacles, robot shapes, alignment, sensor noise, interference, possession mishaps or pickup uncertainty.
5. **Autonomous line & AWP:** Opposing-side exclusion is approximated by left/right x-half. Perimeter contact is a simplified center-position threshold. Violation effects and manual intervention are not simulated.
6. **Midfield:** Simplified square footprint around center; robot body geometry not represented. Robot positioning is updated at action completion.
7. **Supply:** Six configurable stock piles, no physical Match Load mechanics; each friendly robot starts with one allied Pin as a preload.
8. **Opponents:** Scripted baselines or an earlier PPO snapshot; not necessarily representative of actual teams.
9. **Score events:** Score changes are computed during simulation for RL feedback even though official match scores are judged after the match. Terminal score reflects the final abstract field state; no physical referee delay.

Because points depend on the assumed rules, **model win rates from this simulator cannot be treated as predictions of real match outcomes**. Prioritize validating score interactions and changing model assumptions after teammate integration. Always include the config and simulator version with research results.

## Authoritative sources

- https://manuals.vex.com/v5/override/v2.1/game/
- https://manuals.vex.com/v5/override/v2.1/vex-u/
- https://www.vexrobotics.com/v5/competition
