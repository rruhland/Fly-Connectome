"""Honor the public rate and remaining motor excursion across the entire strike plan."""
import numpy as np
from scipy.optimize import minimize

from .tt6_body_control import MOTOR_BUDGET
from .tt8_service_drive import ServiceDriveServo
from .tt9_precontact_registration import RegistrationPlayer


class FeasibleServo(ServiceDriveServo):
    def _solve_plan(self, normal, rhs):
        n = self.fit.action_size
        raw = np.linalg.solve(normal, rhs)
        rates = raw.reshape(-1,n)
        predicted = self.requested + rates.cumsum(axis=0)
        if np.max(abs(rates)) <= 1. and np.max(abs(predicted)) <= MOTOR_BUDGET:
            return rates[0]
        if not hasattr(self,'cumulative'):
            self.cumulative = np.kron(np.tri(len(rates)),np.eye(n))
            self.inequality = np.vstack((self.cumulative,-self.cumulative))
        # The old solver clips only the current action. Use its sequentially
        # feasible continuation as the initial solution of the constrained QP.
        initial = []
        state = self.requested.copy()
        for rate in rates:
            action = np.clip(rate,np.maximum(-1.,-MOTOR_BUDGET-state),np.minimum(1.,MOTOR_BUDGET-state))
            initial.append(action)
            state += action
        start = np.ravel(initial)
        lows = np.tile(-MOTOR_BUDGET-self.requested,len(rates))
        highs = np.tile(MOTOR_BUDGET-self.requested,len(rates))
        bounds = np.r_[-lows,highs]
        # Normalize for the solver's absolute objective tolerance.
        scale = max(float(np.max(np.diag(normal))),1e-6)
        h, r = normal/scale, rhs/scale
        def objective(x):
            grad = h @ x - r
            return .5 * x @ (grad-r), grad
        result = minimize(objective,start,jac=True,method='SLSQP',bounds=[(-1.,1.)]*len(raw),
                          constraints=[dict(type='ineq',fun=lambda x: bounds+self.inequality@x,
                                            jac=lambda x:self.inequality)],
                          options=dict(ftol=1e-7,maxiter=40))
        feasible = np.min(bounds+self.inequality@result.x) >= -1e-7
        better = result.fun <= objective(start)[0]
        plan = result.x if feasible and better else start
        return np.clip(plan[:n], np.maximum(-1.,-MOTOR_BUDGET-self.requested),
                       np.minimum(1.,MOTOR_BUDGET-self.requested))


class FeasiblePlayer(RegistrationPlayer):
    def _make_servo(self):
        return FeasibleServo(self.adapter,self.timing_memory,learn=self.adjust)
