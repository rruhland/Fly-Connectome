"""Compare saved visual forecasts to referee events, outside the controller."""
import argparse
from pathlib import Path
from statistics import median

from .tt4_transfer_trial import _hash, _read, _write
from .tt5_player import FRAME_DT


def inspect(point):
    choices = point['reply_window']['choices']
    if not choices:
        return None
    choice, = choices
    forecast = choice['selected']
    first = point['summary']['first_hit']
    events = point['events']
    after_reply = point['identical_through_rival_contact']
    hit = next((event for event in events if event['kind'] == 'contact' and event['side'] == 1
                and event['step'] > after_reply), None)
    if (None if hit is None else hit['step']) != first:
        raise ValueError('point contact does not match the referee summary')
    flight = []
    if hit is not None:
        for event in events[events.index(hit) + 1:]:
            if event['kind'] == 'contact':
                break
            flight.append(event)
    bounce = next((event['location'][1] for event in flight if event['kind'] == 'bounce'), None)
    legal = any(event['kind'] == 'legal_return' and event['side'] == 1 for event in flight)
    trace = point['optical_trace']
    decision_time = choice['tick'] + forecast['time']
    actual_time = None if hit is None else hit['time'] / FRAME_DT
    views = [row for row in trace if row['blade'] is not None and
             first is not None and first - 2 <= row['tick'] <= first]
    optical_speed = None
    if len(views) >= 2:
        earlier, later = views[-2:]
        optical_speed = ((later['blade'][1] - earlier['blade'][1]) /
                         (later['tick'] - earlier['tick']))
    return dict(seed=point['seed'], phase=point['phase'], depth_offset=choice['depth_offset'],
                decision_tick=choice['tick'], predicted_face_tick=decision_time,
                physical_contact_tick=actual_time, forecast_contact_depth=forecast['ball'][1],
                physical_contact_depth=None if hit is None else hit['ball'][1],
                predicted_forward_speed=forecast['blade_velocity'][1],
                optical_forward_speed_near_contact=optical_speed,
                predicted_clearance=forecast['clearance'], predicted_bounce=forecast['bounce_depth'],
                predicted_legal=forecast['clearance'] > 0. and forecast['bounce_depth'] is not None,
                physical_first_bounce=bounce, physical_legal=legal)


def audit(paths):
    rows = []
    for path in paths:
        receipt = _read(path)
        if not receipt['paired']:
            raise ValueError('audit expects a paired evaluation')
        for point in receipt['sections']['window']['points']:
            row = inspect(point)
            if row is not None:
                rows.append(dict(installation=receipt['installation_seed'], **row))
    contacted = [row for row in rows if row['physical_contact_tick'] is not None]
    measured = [row for row in contacted if row['optical_forward_speed_near_contact'] is not None]
    time_errors = [abs(row['predicted_face_tick'] - row['physical_contact_tick']) for row in contacted]
    depth_errors = [abs(row['forecast_contact_depth'] - row['physical_contact_depth']) for row in contacted]
    speed_errors = [abs(row['predicted_forward_speed'] - row['optical_forward_speed_near_contact'])
                    for row in measured]
    return dict(schema='tt10-window-forecast-audit-v1', source_sha256={p.name: _hash(p) for p in paths},
                selections=len(rows), contacts=len(contacted), rows=rows,
                predicted_legal=sum(row['predicted_legal'] for row in rows),
                actual_legal=sum(row['physical_legal'] for row in rows),
                correctly_predicted_legal=sum(row['predicted_legal'] and row['physical_legal'] for row in rows),
                predicted_bounces_within_far_half=sum(row['predicted_bounce'] is not None for row in rows),
                actual_table_bounces=sum(row['physical_first_bounce'] is not None for row in rows),
                median_contact_time_error_frames=median(time_errors) if time_errors else None,
                median_contact_depth_error_m=median(depth_errors) if depth_errors else None,
                contact_optical_speed_estimates=len(measured),
                median_absolute_speed_error_m_per_frame=median(speed_errors) if speed_errors else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('receipts', nargs='+', type=Path)
    args = p.parse_args()
    _write(args.output, audit(args.receipts))
