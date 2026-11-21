"""Commission one stationary stereo installation using as few camera probes as needed."""
from .tt4_transfer import ZERO, calibration_commands
from .tt5_player_body import fit_arm
from .tt5_vision import hall_tracker_factory


def commission(wiring, rig, schedule, horizon):
    trials = []
    histories, paid = [], []
    last_error = 'no calibration attempted'
    for index, seed in enumerate(schedule['calibration']):
        images = wiring.reset(seed=seed)
        history = [(ZERO,images)]
        for command in calibration_commands(schedule['player_seed']+index, horizon):
            images, _, term, trunc, _ = wiring.step(command)
            history.append((command,images))
            if term or trunc:
                break
        histories.append(history)
        paid.append(dict(seed=seed, steps=len(history)-1, frames=len(history),
                         actions=[a.tolist() for a,_ in history[1:]]))
        count = len(paid)
        if count not in (3,5,len(schedule['calibration'])):
            continue
        try:
            factory = hall_tracker_factory(rig,histories)
            model = fit_arm(histories,factory)
            if count < len(schedule['calibration']) and (model.error > .065 or model.gains[3] == 0):
                raise ValueError('require a better position fit and a signed wrist response to stop early')
        except ValueError as e:
            last_error = str(e)
            trials.append(dict(probes=count,error=last_error))
            continue
        trials.append(dict(probes=count,median_fit_error=model.error))
        return model, factory, paid, trials, None
    return None, None, paid, trials, last_error
