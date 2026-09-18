"""Frozen, held-out comparisons. Scientific conclusions are reported, not assumed."""
from .training import load_checkpoint


def evaluate(checkpoint, seeds, steps, control='learned', device='cpu'):
    if steps <= 0:
        raise ValueError("evaluation requires positive steps")
    trainer = load_checkpoint(checkpoint, device=device, evaluation=True, seeds=seeds)
    if set(seeds) & set(trainer.training_seeds):
        raise ValueError("evaluation seeds overlap training")
    metrics = trainer.run(steps, control)
    samples = max(1, metrics.pop('prediction_samples'))
    metrics['local_current_prediction_mse'] = metrics.pop('prediction_squared_error') / samples
    metrics['local_current_persistence_mse'] = metrics.pop('persistence_squared_error') / samples
    event_samples = max(1, metrics.pop('event_samples'))
    metrics['sensory_event_prediction_mse'] = metrics.pop('event_prediction_squared_error') / event_samples
    metrics['sensory_event_persistence_mse'] = metrics.pop('event_persistence_squared_error') / event_samples
    metrics['sensory_event_zero_mse'] = trainer.event_zero_error.item() / event_samples
    metrics['mean_rate_hz'] = metrics['spikes'] / (steps * trainer.environment.config.dt *
                                                len(seeds) * trainer.network.n)
    types = trainer.retina.spec['cell_types']
    counts = trainer.evaluation_spike_counts.sum(0).cpu().tolist()
    populations = {}
    for label, count in zip(types, counts):
        populations[label] = populations.get(label, 0) + count
    return dict(metrics=metrics, seeds=seeds, steps=steps, control=control, hit_shaping=0.,
                population_spikes=populations,
                event_target_encoding=('signed-lamina-contrast' if 'contrast' in trainer.retina.spec['injection'].values()
                                       else 'binary-polarity-routed-events'),
                graph_sha256=trainer.network.graph.identity(), dataset=trainer.manifest.get('dataset'))


def compare(learned_checkpoint, initial_checkpoint, seeds, steps, device='cpu'):
    learned = load_checkpoint(learned_checkpoint, device=device, evaluation=True, seeds=seeds)
    initial = load_checkpoint(initial_checkpoint, device=device, evaluation=True, seeds=seeds)
    if learned.network.graph.identity() != initial.network.graph.identity():
        raise ValueError("baseline comparisons require identical topology and initialization scale")
    if learned.environment.config != initial.environment.config:
        raise ValueError("baseline comparisons require identical physics")
    if (learned.config != initial.config or learned.network.config != initial.network.config or
            learned.retina.spec != initial.retina.spec or
            not learned.network.delays.equal(initial.network.delays) or
            not learned.network.pathways.equal(initial.network.pathways)):
        raise ValueError("baseline comparisons require identical dynamics and sensory mapping")
    visual = learned.config.stage == 'M1A'
    del learned, initial
    comparisons = (('learned', learned_checkpoint, 'scripted'), ('frozen', initial_checkpoint, 'scripted')) if visual else (
        ('learned', learned_checkpoint, 'learned'), ('frozen', initial_checkpoint, 'learned'),
        ('random', initial_checkpoint, 'random'), ('scripted', initial_checkpoint, 'scripted'))
    return {name: evaluate(path, seeds, steps, mode, device) for name, path, mode in comparisons}
