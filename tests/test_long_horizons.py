import importlib.util
from pathlib import Path
import torch


def module(monkeypatch):
    folder=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('long_horizons',folder/'audit_long_horizons.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_next_event_uses_future_only_and_censors_tail(monkeypatch):
    m=module(monkeypatch)
    targets=torch.tensor([1.,0.,-1.,0.,0.,1.]).view(6,1,1)
    labels,wait=m.next_events(targets,2)
    assert labels.flatten().tolist()==[-1.,-1.,0.,1.]
    assert wait.flatten().tolist()==[2,1,0,2]


def test_fixed_horizon_reads_only_earlier_neural_activity(monkeypatch):
    m=module(monkeypatch)
    y=torch.tensor([0.,1.,-1.,1.,-1.,1.]).view(6,1,1)
    timeline=torch.arange(13,dtype=torch.float32).view(13,1,1)
    result=m.fixed_horizons(y,timeline,torch.tensor([True]),torch.tensor([True]),2,.01,(30,))
    # 3 ticks of lead: target frame f uses stored timeline[2*f-2].
    frames=result['target_frames']
    expected=timeline[torch.tensor(frames)*2-2].double().square().mean()
    assert result['scores']['30']['model']['all']['prediction_power']==float(expected)
    assert result['scores']['30']['actual_lead_ms']==30.
    known=y[(torch.tensor(frames)*2-3)//2]
    expected_persistence=(y[frames].double()-known.double()).square().mean()
    assert result['scores']['30']['frame_persistence']['all']['model_mse']==float(expected_persistence)


def test_next_event_controls_include_quiet_windows_and_only_past_polarity(monkeypatch):
    m=module(monkeypatch)
    targets=torch.zeros(20,1,1)
    targets[0]=1
    targets[2]=-1
    result=m.diagnose(targets,torch.zeros(41,1,1),torch.tensor([True]),torch.tensor([True]),2,.01)['next_event']
    assert result['decision_frames']==15
    assert result['windows_with_event']==2
    assert result['windows_without_event']==13
    # Both future-event windows have observed +1; its opposite predicts -1 exactly.
    assert result['scores']['opposite_last_polarity']['on_supported']['model_mse']==0
    assert result['scores']['opposite_last_polarity']['quiet']['model_mse']==1
    assert result['scores']['model']['all']['model_mse']==2/15
