"""Inference-only empty-input correction for the frozen Ogma-prior endpoint."""
import numpy as np
from pong_ogma_prior import OgmaRuntime

CONTRACT='empty-current1792-exact40069-override-after-temporal-step-v1'


class EmptyCurrentGuard(OgmaRuntime):
    def raw_query(self,indices,values,learn_representation=False,learn_actor=False):
        if learn_representation or learn_actor:raise ValueError('frozen inference correction only')
        # Advance every temporal/predictive buffer before considering the override.
        raw=super().raw_query(indices,values,False,False)
        present=np.asarray(indices)<1792
        empty=not np.any(np.asarray(values)[present]!=0)
        result=dict(raw,guard_contract=CONTRACT,empty_current_fallback=empty,
                    unguarded_action=raw['action'],unguarded_q=raw['q'])
        if empty:
            action,q=self.baseline.choose(indices,values)
            result.update(action=int(action),greedy=int(action),q=np.asarray(q).tolist())
        return result
