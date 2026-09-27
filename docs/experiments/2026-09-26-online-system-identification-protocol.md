# Entity-local online system identification

Close endpoint-prototype, local-linear-prototype and learned-competence variants:
their aggregate gains do not remove the straight-motion regression. Keep the
statistical observation state and learned contextual representation.

Test a different organization: each persistent entity owns a small online
autoregressive transition, updated from its own actual observations. Four past
displacements drive a shared two-axis linear recurrence. Complex-valued temporal
coefficients represent the generic local operations I and 90-degree rotation J;
no direction, speed, object category or trajectory family is supplied. Local
correlation updates solve eight coefficients with a unit Gaussian ridge prior
centered on the recent-displacement mean. Reset the operator on a new identity,
not when the scene's motion changes. Never train on extrapolated positions.

Compare learning with its frozen prior and all fixed controls on the registered
noisy generic transfer grid. Each forecast rolls the currently learned transition
forward without future observations. No coefficient/lag/regularization sweep.
This tests whether pooling different entities into one scalar endpoint memory was
the wrong credit structure. It is an engineered local linear dynamical state,
not a claim of a new biological circuit.
