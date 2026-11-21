# Continuous serve search on offset

Seven rounds of live practice compared the incumbent with four mirrored perturbations on the same toss within each round. The six search dimensions were contact height, blade inclination, forward and vertical motion, timing lead, and strike depth. The search selected updates from optical travel and bounce estimates and the public point reward. Serve legality and contact events were recorded for evaluation.

Thirty of the 35 practice attempts were legal across the seven tosses. The selected controller targeted a contact height of 0.449 m above the table, a blade inclination of 0.356 rad, and blade travel of 0.109 m forward and 0.040 m down per frame. It advanced the stroke by 1.75 frames and shifted the strike target 0.047 m toward the net.

| Controller | Legal serves | Lets | Points won |
| --- | ---: | ---: | ---: |
| Frozen searched stroke | 2/10 | 1 | 2/10 |
| Fixed velocity stroke | 7/10 | 0 | 6/10 |

Both controllers faced the same ten fresh tosses, evenly split between soft and mixed command-delay conditions. The searched stroke missed three tosses outright. The fixed velocity stroke landed five soft and two mixed serves legally. The alternating game required the searched stroke to improve held legality. The held result did not meet that requirement.

All 55 action tapes reproduced the applied commands, physical events, and point results in replay. [Trial receipt](service-search-offset.json.gz) · [Selected stroke checkpoint](service-search-offset-player.json.gz)
