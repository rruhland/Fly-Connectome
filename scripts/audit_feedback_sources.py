"""Frozen source activity behind a population's measured predictive inputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


def restore_c2_inputs(network,reference,types):
    assert network.graph.identity()==reference.graph.identity()
    assert network.config==reference.config
    assert torch.equal(network.pathways,reference.pathways)
    assert torch.equal(network.delays,reference.delays)
    c2=torch.tensor([t=='C2' for t in types])
    mask=(network.pathways==1)&c2[network.post]
    network.magnitudes[mask]=reference.magnitudes[mask]
    return int(mask.sum())


class SourceAudit:
    def __init__(self,network,types,population):
        self.net,self.types=network,types
        target=torch.tensor([t==population for t in types])
        self.targets=target.nonzero().flatten()
        self.edges=((network.pathways==1)&target[network.post]).nonzero().flatten()
        self.sources=network.pre[self.edges].unique()
        self.lookup=torch.full((network.e,),-1,dtype=torch.long)
        self.lookup[self.edges]=torch.arange(len(self.edges))
        self.trace=torch.zeros(network.batch,len(self.edges))
        self.energy=torch.zeros(len(self.edges),dtype=torch.float64)
        self.spikes=torch.zeros(len(self.sources),dtype=torch.int64)
        self.currents=torch.zeros(6,len(self.sources),dtype=torch.float64)
        self.margin=torch.full((len(self.sources),),-torch.inf)
        self.ticks=0
        self.error=0.

    def observe(self,activity,*,measure):
        n,s=self.net,self.sources
        if measure:
            # Match the preceding-tick physical trace used by audit_local_signals.
            self.energy.add_(self.trace.double().square().sum(0))
            self.spikes.add_(activity.spikes[:,s].sum(0))
            for i,current in enumerate((n.feedforward_current,n.predictive_current,n.sensory_state)):
                values=current[:,s].double()
                self.currents[i].add_(values.sum(0))
                self.currents[i+3].add_(values.abs().sum(0))
            # Recorded after spike reset: useful for silent sources, not spike overshoot.
            self.margin=torch.maximum(self.margin,(n.voltage[:,s]-n.config.threshold-n.adaptation[:,s]).max(0).values)
            self.ticks+=1
        self.trace.mul_(n.current_decay[n.post[self.edges]])
        local=self.lookup[activity.arrival_edges]
        keep=local>=0
        self.trace.index_put_((activity.arrival_environments[keep],local[keep]),
                             n.signs[activity.arrival_edges[keep]],accumulate=True)
        reconstructed=torch.zeros_like(n.voltage)
        reconstructed.index_add_(1,n.post[self.edges],self.trace*n.magnitudes[self.edges])
        if len(self.targets):
            self.error=max(self.error,float((reconstructed[:,self.targets]-activity.predicted[:,self.targets]).abs().max()))

    def report(self):
        n=self.net
        denominator=self.ticks*n.batch
        power=self.energy/denominator
        means=self.currents/denominator
        rows=[]
        for i,source in enumerate(self.sources.tolist()):
            mask=n.pre[self.edges]==source
            rows.append(dict(body_id=int(n.graph.body_ids[source]),cell_type=self.types[source],
                feedback_edges=int(mask.sum()),active_feedback_edges=int((power[mask]>1e-12).sum()),
                spikes=int(self.spikes[i]),rest_current=float(n.rest_current[source]),
                mean_feedforward=float(means[0,i]),mean_predictive=float(means[1,i]),mean_sensory=float(means[2,i]),
                mean_abs_feedforward=float(means[3,i]),mean_abs_predictive=float(means[4,i]),
                mean_abs_sensory=float(means[5,i]),maximum_poststep_margin=float(self.margin[i]),
                outgoing_weight_sum=float(n.magnitudes[self.edges[mask]].double().sum())))
        classes={}
        for label in sorted({r['cell_type'] for r in rows}):
            selected=[r for r in rows if r['cell_type']==label]
            classes[label]=dict(sources=len(selected),spiking_sources=sum(r['spikes']>0 for r in selected),
                **{key:sum(r[key] for r in selected) for key in ('feedback_edges','active_feedback_edges','spikes')},
                **{key:sum(r[key] for r in selected)/len(selected) for key in
                   ('mean_feedforward','mean_predictive','mean_sensory','mean_abs_feedforward','mean_abs_predictive')})
        return dict(measured_ticks=self.ticks,maximum_reconstruction_error=self.error,classes=classes,sources=rows)


@torch.no_grad()
def audit(checkpoint,steps,seeds,population='L3',restore_c2_from=None):
    if steps<2:
        raise ValueError('at least two frames required')
    before=checksum(checkpoint)
    model=load_checkpoint(checkpoint,evaluation=True,seeds=seeds,warmup=False)
    assert model.plasticity is None
    restoration=None
    if restore_c2_from:
        reference_hash=checksum(restore_c2_from)
        reference=load_checkpoint(restore_c2_from,evaluation=True,seeds=seeds,warmup=False)
        assert model.config==reference.config and model.learning_config==reference.learning_config
        assert model.environment.config==reference.environment.config and model.retina.spec==reference.retina.spec
        restored=restore_c2_inputs(model.network,reference.network,model.retina.spec['cell_types'])
        restoration=dict(reference_sha256=reference_hash,restored_edges=restored,
            interpretation='Evaluation-only hybrid: restore existing predictive magnitudes entering C2 before warmup. Not a trained checkpoint or acceptance result.')
        del reference
    recorder=SourceAudit(model.network,model.retina.spec['cell_types'],population)
    original=model.network.step
    def record(*args,**kwargs):
        activity=original(*args,**kwargs)
        recorder.observe(activity,measure=model.step_index>0)
        return activity
    model.network.step=record
    model.warmup()
    model.run(steps,'scripted')
    result=recorder.report()
    assert result['maximum_reconstruction_error']<1e-5
    assert checksum(checkpoint)==before
    if restore_c2_from:
        assert checksum(restore_c2_from)==reference_hash
    configuration=dict(neurons=asdict(model.network.config),learning=asdict(model.learning_config),
        run=asdict(model.config),physics=asdict(model.environment.config),retina=model.retina.spec)
    result.update(checkpoint_sha256=before,graph_sha256=model.network.graph.identity(),
        configuration_sha256=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
        training_steps=model.training_step,frames=steps,seeds=seeds,population=population,
        restoration=restoration,
        source_statistic_frames=steps-1,event_metric_frames=steps,
        sensory_event_mse=model.metrics['event_prediction_squared_error']/model.metrics['event_samples'],
        sensory_zero_mse=float(model.event_zero_error)/model.metrics['event_samples'],
        interpretation='Frozen tensor reference; physical traces include warmup. Source/trace statistics exclude the first frame; sensory event metrics include all frames. Per-source means average batch and ticks, class means average sources. Margin is post-reset voltage minus adapted threshold, not spike overshoot. Source checkpoints unchanged; any evaluation-only weight restoration is identified separately. Final seeds unused.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seeds',default='1101,1102')
    parser.add_argument('--population',default='L3')
    parser.add_argument('--restore-c2-from')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=audit(args.checkpoint,args.steps,[int(s) for s in args.seeds.split(',')],args.population,args.restore_c2_from)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['classes']),flush=True)
