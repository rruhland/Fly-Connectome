"""Online production vision plus transport latency; excludes sensor acquisition."""

import json
import time
from pathlib import Path

import torch

from fly_connectome.vision import VisualStateEncoder, load_default


@torch.no_grad()
def main(*, out=Path('docs/experiments/2026-09-27-visual-interface-timing.json'), learn=True):
    torch.set_num_threads(1)
    result = dict(threads=1, sensor='64x64 every-sample frames plus events', online=learn, arms={})
    for count in (2, 8):
        model, encoder = load_default(), VisualStateEncoder(64, 64, .02)
        def memory():
            banks = {f'{h}/{name}': bank for h, m in model.dynamics.items()
                     for name, bank in ((('short', m.short), ('long', m.long))
                                        if hasattr(m, 'long') else (('legacy', m),))}
            return dict(examples={name: len(bank.keys) for name, bank in banks.items()},
                        bank_tensor_bytes=sum(t.numel()*t.element_size() for bank in banks.values()
                                              for t in (bank.keys, bank.values)))
        initial_memory = memory()
        previous = torch.zeros(64, 64)
        full_times, encoder_times, active, detected = [], [], [], []
        for t in range(240):
            image = torch.zeros(64, 64)
            for index in range(count):
                y, x = 4+index*7, 5+int((t+index*3) % 80 if (t+index*3) % 80 < 40 else 79-(t+index*3) % 80)
                image[y:y+3, x:x+3] = 1
            events = torch.stack(((previous-image).clamp(min=0), (image-previous).clamp(min=0)))
            previous = image
            start = time.perf_counter()
            state = model.step(events, image, learn=learn)
            middle = time.perf_counter()
            output = encoder.encode(state, image)
            end = time.perf_counter()
            if t >= 40:
                full_times.append((end-start)*1000)
                encoder_times.append((end-middle)*1000)
                active.append(len(output['indices']))
                detected.append(len(state['entities']))
        def summary(values):
            return dict(zip(('p50', 'p95'), torch.tensor(values, dtype=torch.float64).quantile(torch.tensor([.5, .95], dtype=torch.float64)).tolist()))
        result['arms'][str(count)] = dict(samples=len(active), total_ms=summary(full_times),
            encoder_ms=summary(encoder_times), active_features=summary(active),
            detected_entities=summary(detected), dimension=output['dimension'],
            initial_memory=initial_memory, final_memory=memory())
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
