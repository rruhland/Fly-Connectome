# Second returns on independent recorded trajectories

On four unperturbed measured launches, the full-size SO-101 simulation landed two second returns with ongoing stereo retargeting. The frozen recall control landed zero. A separate planner that simulated candidate strokes from the same RGB observations landed three. The returning arm made second contacts on 3/4 points with retargeting, 1/4 with frozen recall and 4/4 with planning. [Original-recording points and both joint-rate tapes](reply-recorded.json.gz)

| Recorded donor | Frozen recall | Stereo retargeting | RGB-state planning | True-state feasibility probe |
| --- | --- | --- | --- | --- |
| 14431 | Miss | Contact | Legal | Legal |
| 10597 | Miss | Miss | Contact | 83 contacts in 140 probes |
| 6029 | Miss | Legal | Legal | Legal |
| 3463 | Contact | Legal | Legal | Legal |

The four points were all existing unmodified recordings that had a legal first block and legal reply from the simulated opponent. The second-return course reused their archived first-block commands. A new stereo image sequence determined the handoff in every mode. The same live opponent acted throughout, including after a successful second stroke. On donor 3463 it returned the stereo-retargeted second shot, producing four consecutive legal returns after the feed. It also returned second shots from the planner on donors 6029 and 3463. Both arms' issued commands reproduced every referee event across all 12 points. [Opening roster](../tt25/recorded-live-opponent.json.gz) · [Verified point records](reply-recorded.json.gz)

The frozen memory contained seven demonstrated strokes from donors 2999, 11953 and 13009. Both mounted variants of a recorded donor stayed on the same side of the split. The neutral-mount replication transported the same memory for every policy by preserving the demonstrated contact pose's paddle center and normal. [Demonstration trajectories and provenance](reply-traces.json.gz) · [Original-recording runner](../../scripts/experimental/tt27_reply_recorded.py)

The camera controller tracked ordinary orange pixels from the fixed stereo pair. It read the ball reversing toward the learner, fit recent flight samples and refreshed its predicted position and arrival time at the demonstration's interception plane as more frames arrived. It corrected the joint goal with the calibrated physical arm chain. The frozen control used the same initial visual match and then issued the training stroke as stored. In donor-grouped development, the ongoing fixed-example correction made six second contacts and two legal returns on 12 physical points. Frozen recall made zero contacts and one-shot correction made six contacts and one legal return. Development used archived RGB-derived ball observations and stopped feeding those observations after a new contact. [Retargeting code](../../scripts/experimental/tt27_reply.py) · [Development controls](reply-development.json.gz)

The planning comparison used a separate simulated arm initialized from the fitted RGB ball state, an assumed spin, measured learner joints and locally accumulated motor commands. It tested small changes around learned strokes and executed a selected stroke in the scoring arena. The feasibility probe independently tested candidate joint motions on the actual simulation branch using its full state. It found legal strokes on three of the four original recordings in 295 probes. The finite search on 10597 establishes physical contact on that reply. [Planning code](../../scripts/experimental/tt27_reply_shadow.py) · [Feasibility probes](reply-recorded-feasibility.json.gz)

At mounts −0.22 and +0.22 radians, the paired course used 12 live reply opportunities from excluded donors 4104 and 10542. The reactive planner landed six second returns across both donors. Ongoing retargeting increased second contacts from 3/12 to 7/12. The ten opportunities with launch indices at least 12 had zero legal second returns under frozen recall, one each under ongoing feedback and one-shot correction, and five under planning. On held donor 4104, mount +0.22 and launch 21, ongoing retargeting landed the second shot and frozen recall missed. The privileged feasibility probe found legal strokes in 9/12 mounted cases after 677 attempts. [Mounted points and both joint-rate tapes](reply-held.json.gz) · [Mounted feasibility probes](reply-held-feasibility.json.gz)

| Mounted controller | Second contacts | Legal second returns |
| --- | --- | --- |
| Frozen recall | 3/12 | 1/12 |
| One-shot stereo correction | 5/12 | 2/12 |
| Ongoing stereo correction | 7/12 | 1/12 |
| RGB-state planning | 7/12 | 6/12 |

In a separate eight-opportunity course on new recorded donors, scalar-reward selection landed zero second returns and the frozen ongoing camera controller landed two. The selector updated its choice among four visual controllers after each point using only its own binary legal-return reward. It made two second contacts and the frozen controller made five. The two successful frozen points were assigned different visual controllers by the learner after earlier misses. Ten first blocks generated the eight legal physical replies used here. Each unique branch replayed its referee events from the two arms' commands. [Chronological point and reward records](reply-online.json.gz) · [Online selector](../../scripts/experimental/tt27_reply_online.py)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply collect
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply development
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply evaluate
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply_recorded held-feasibility
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply_recorded path/to/recorded-feeds.json recorded
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply_recorded path/to/recorded-feeds.json recorded-feasibility
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_reply_online path/to/recorded-feeds.json
```
