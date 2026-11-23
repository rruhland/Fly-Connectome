"""Use one wrist controller and its learned actuator memories across physical games."""
from pathlib import Path

import numpy as np

from .tt24_autonomous_drive import MotorMemory
from .tt24_carried_wrist import BUNDLE
from .tt24_motor_bank import I, MotorBank, decoder
from .tt24_tool_equilibrium import ToolEquilibrium


class ContinualWrist:
    """Camera and encoder inputs choose the reference and signal when motor inspection is needed."""
    def __init__(self,checkpoint=BUNDLE):
        self.checkpoint=Path(checkpoint)
        self.monitor=MotorMemory()
        self.bank=MotorBank()
        self.last_eye=None
        with np.load(checkpoint,allow_pickle=False) as d:
            limit=float(d["live_workspace_limit"]) if "live_workspace_limit" in d else .0557
            if "live_schema" in d:
                if d["live_schema"].item()!="tt24-continual-wrist-1":
                    raise ValueError("unsupported motor memory")
                self.bank=MotorBank(d["live_maps"])
                selected=int(d["live_selected"])
                if selected>=0:
                    self.bank.selected=selected
                    self.bank.inverse=decoder(self.bank.maps[selected])
                self.monitor.pending=bool(d["live_pending"])
                eye=int(d["live_last_eye"])
                self.last_eye=eye if eye>=0 else None
        self.policy=ToolEquilibrium(checkpoint,limit)
        self.policy.mode="workspace_local"

    def open_release(self,observation):
        """Check the opening wrist image. A positive result requests a contact-free setup interval."""
        self.policy.begin_release(observation)
        eye=self.policy.calibration_index
        check=self.monitor.pending or (self.last_eye is not None and self.last_eye!=eye)
        self.last_eye=eye
        return check

    def inspect(self,joints,send_and_read_joints,seed):
        """Send bounded rates to this arm, measure its encoder response, and finish near its starting target."""
        requests=[]
        def send(command):
            requests.append(command.copy())
            return send_and_read_joints(command)
        report=self.bank.connect(joints,send,seed,measured_settling=True)
        net=np.sum(requests,axis=0)
        if np.linalg.norm(net)>1.e-7:
            last=None
            for _ in range(4):
                last=send(-net/4)
            stable=0
            for _ in range(30):
                current=send(np.zeros(5))
                stable=stable+1 if np.max(abs(current-last))<.00045 else 0
                last=current
                if stable>=3:
                    break
        self.monitor.pending=False
        return report|dict(total_ticks=len(requests))

    def begin_play(self,observation):
        """Start feedback from the physical posture after any setup motion."""
        self.policy.begin_release(observation)
        self.monitor.current=I if self.bank.selected is None else self.bank.maps[self.bank.selected]
        self.monitor.start(observation["joints"])

    def act(self,observation):
        intended=self.policy.act(observation)
        return np.clip(intended,-1.,1.) if self.bank.selected is None else self.bank.apply(intended)

    def observe(self,requested,next_observation):
        """Log the requested rate and the next public encoder reading after the step."""
        return self.monitor.observe(requested,next_observation["joints"])

    def save_between_releases(self,path):
        """Save the complete wrist readout, optical calibration and learned interface bank."""
        with np.load(self.checkpoint,allow_pickle=False) as archive:
            data={k:archive[k] for k in archive if not k.startswith("live_")}
        data.update(live_schema=np.array("tt24-continual-wrist-1"),
                    live_maps=np.array(self.bank.maps),
                    live_selected=np.array(-1 if self.bank.selected is None else self.bank.selected),
                    live_last_eye=np.array(-1 if self.last_eye is None else self.last_eye),
                    live_pending=np.array(self.monitor.pending),live_workspace_limit=np.array(self.policy.workspace_limit))
        np.savez_compressed(path,**data)
