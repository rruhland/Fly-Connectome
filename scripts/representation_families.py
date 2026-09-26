"""Seven opt-in local visual representation principles, without gradients."""

import math

import torch
import torch.nn.functional as F


OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))


def _random(shape, seed, scale=.1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator)*scale


def _shift(value, dy, dx):
    shifted = torch.roll(value, (-dy, -dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., -dy:, :] = 0
    elif dy < 0:
        shifted[..., :-dy, :] = 0
    if dx > 0:
        shifted[..., -dx:] = 0
    elif dx < 0:
        shifted[..., :-dx] = 0
    return shifted


def _compete(activity, winners=2):
    activity = activity.clamp(min=0.)
    threshold = activity.topk(winners, dim=0).values[-1]
    activity = activity*(activity >= threshold)
    return activity/activity.sum(0, keepdim=True).clamp(min=1.)


class SlowFeature:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.kernel = _random((8, 6, 3, 3), 11)
        self.bias = torch.zeros(8)
        self.reset_state()

    def learned_tensors(self):
        return (self.kernel, self.bias)

    def reset_state(self):
        self.state = torch.zeros((8, self.height, self.width))
        self.previous_input = None
        self.previous_response = None

    @torch.no_grad()
    def step(self, value, *, learn=False):
        response = _compete(F.conv2d(
            value[None], self.kernel, padding=1)[0]-self.bias[:, None, None])
        if learn and self.previous_input is not None:
            pre_change = value-self.previous_input
            post_change = response-self.previous_response
            patches = F.unfold(pre_change[None], 3, padding=1)[0]
            update = -post_change.reshape(8, -1) @ patches.T
            exposure = post_change.abs().sum((1, 2)).clamp(min=1.)
            self.kernel.add_(.02*(update/exposure[:, None]).reshape_as(
                self.kernel)).clamp_(-.5, .5)
            # A weak local variance term prevents a constant slow code.
            patches = F.unfold(value[None], 3, padding=1)[0]
            retained = response.reshape(8, -1) @ patches.T
            self.kernel.add_(.002*(retained/response.sum((1, 2)).clamp(
                min=1.)[:, None]).reshape_as(self.kernel)).clamp_(-.5, .5)
            self.bias.add_(.001*(response.mean((1, 2))-.02)).clamp_(-.2, .2)
        self.state = .75*self.state+.25*response
        self.previous_input = value.clone()
        self.previous_response = response.clone()
        return self.state.clone()


class CommonFate:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.link_prior = torch.zeros(8)
        self.reset_state()

    def learned_tensors(self):
        return (self.link_prior,)

    def reset_state(self):
        self.previous = torch.zeros((6, self.height, self.width))
        self.activity = torch.zeros_like(self.previous)
        self.links = torch.zeros((8, self.height, self.width))

    @torch.no_grad()
    def step(self, value, *, learn=False):
        change = value-self.previous
        agreements = []
        for dy, dx in OFFSETS:
            neighbor = _shift(change, dy, dx)
            agreement = ((change*neighbor).sum(0)/(
                change.square().sum(0).sqrt()*neighbor.square().sum(0).sqrt()
                ).clamp(min=1e-6)).clamp(0., 1.)
            agreements.append(agreement)
        agreement = torch.stack(agreements)
        if learn:
            self.link_prior.lerp_(agreement.mean((1, 2)), .02)
        self.links = .8*self.links+.2*agreement*(.5+.5*self.link_prior[:, None, None])
        grouped = torch.zeros_like(value)
        for index, (dy, dx) in enumerate(OFFSETS):
            grouped += self.links[index][None]*_shift(value, dy, dx)
        self.activity = .55*self.activity+.45*(value+.125*grouped)
        self.previous = value.clone()
        return torch.cat((self.activity, self.links)).clone()


class SparseReconstruction:
    def __init__(self, *, height=32, width=64):
        self.dictionary = _random((8, 6, 3, 3), 23)

    def learned_tensors(self):
        return (self.dictionary,)

    def reset_state(self):
        pass

    @torch.no_grad()
    def step(self, value, *, learn=False):
        state = _compete(F.conv2d(value[None], self.dictionary,
                                  padding=1)[0], winners=1)
        if learn and bool(state.any()):
            reconstruction = F.conv_transpose2d(
                state[None], self.dictionary, padding=1)[0]
            residual = value-reconstruction
            patches = F.unfold(residual[None], 3, padding=1)[0]
            update = state.reshape(8, -1) @ patches.T
            exposure = state.sum((1, 2)).clamp(min=1.)
            self.dictionary.add_(.05*(update/exposure[:, None]).reshape_as(
                self.dictionary)).clamp_(-.5, .5)
            norms = self.dictionary.flatten(1).norm(dim=1).clamp(min=1.)
            self.dictionary.div_(norms[:, None, None, None])
        return state.clone()


class AttractorAssembly:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.sensory = _random((8, 6, 3, 3), 31)
        self.recurrent = torch.zeros((8, 8, 3, 3))
        for channel in range(8):
            self.recurrent[channel, channel, 1, 1] = .4
        self.reset_state()

    def learned_tensors(self):
        return (self.sensory, self.recurrent)

    def reset_state(self):
        self.state = torch.zeros((8, self.height, self.width))

    @torch.no_grad()
    def step(self, value, *, learn=False):
        prior = self.state
        drive = F.conv2d(value[None], self.sensory, padding=1)[0]
        drive += F.conv2d(prior[None], self.recurrent, padding=1)[0]
        state = _compete(drive)
        if learn and bool(state.any()):
            patches = F.unfold(value[None], 3, padding=1)[0]
            exposure = state.sum((1, 2)).clamp(min=1.)
            update = state.reshape(8, -1) @ patches.T
            self.sensory.add_(.01*(update/exposure[:, None]).reshape_as(
                self.sensory)).clamp_(-.5, .5)
            prior_patches = F.unfold(prior[None], 3, padding=1)[0]
            update = state.reshape(8, -1) @ prior_patches.T
            self.recurrent.add_(.01*(update/exposure[:, None]).reshape_as(
                self.recurrent)).clamp_(-.2, .5)
        self.state = state.clone()
        return self.state.clone()


class SynchronyBinding:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.coupling = torch.zeros((8, height, width))
        self.reset_state()

    def learned_tensors(self):
        return (self.coupling,)

    def reset_state(self):
        generator = torch.Generator().manual_seed(73)
        self.phase = (2*math.pi*torch.rand(
            (self.height, self.width), generator=generator))
        self.amplitude = torch.zeros_like(self.phase)
        self.previous_energy = torch.zeros_like(self.phase)

    @torch.no_grad()
    def step(self, value, *, learn=False):
        energy = value.abs().sum(0).clamp(0., 1.)
        change = energy-self.previous_energy
        phase_pull = torch.zeros_like(energy)
        for index, (dy, dx) in enumerate(OFFSETS):
            shifted_change = _shift(change, dy, dx)
            agreement = (change*shifted_change).clamp(min=0.)
            if learn:
                self.coupling[index].lerp_(agreement, .02)
            phase_pull += self.coupling[index]*_shift(
                self.amplitude, dy, dx)*torch.sin(
                    _shift(self.phase, dy, dx)-self.phase)
        self.phase = torch.remainder(self.phase+.7+.25*phase_pull, 2*math.pi)
        self.amplitude = .8*self.amplitude+.2*energy
        self.previous_energy = energy.clone()
        # Phase modulates, but does not replace, signed sensory evidence.
        return torch.cat((value*torch.cos(self.phase)[None],
                          value*torch.sin(self.phase)[None],
                          self.amplitude[None])).clone()


class RelationalGraph:
    def __init__(self, *, height=32, width=64):
        assert height % 4 == 0 and width % 4 == 0
        self.height, self.width = height, width
        self.node_height, self.node_width = height//4, width//4
        self.edge = torch.zeros((8, self.node_height, self.node_width))
        self.reset_state()

    def learned_tensors(self):
        return (self.edge,)

    def reset_state(self):
        self.node = torch.zeros((6, self.node_height, self.node_width))
        self.previous = torch.zeros_like(self.node)
        self.active_edge = torch.zeros_like(self.edge)

    @torch.no_grad()
    def step(self, value, *, learn=False):
        sensory = F.avg_pool2d(value[None], 4)[0]
        change = sensory-self.previous
        messages = torch.zeros_like(sensory)
        for index, (dy, dx) in enumerate(OFFSETS):
            neighbor_change = _shift(change, dy, dx)
            coincidence = (16*(change*neighbor_change).sum(0)).clamp(0., 1.)
            if learn:
                self.edge[index].lerp_(coincidence, .02)
            self.active_edge[index] = (.8*self.active_edge[index]+
                                       .2*coincidence)
            messages += (.5*self.edge[index]+.5*self.active_edge[index]) * (
                _shift(self.node, dy, dx))
        self.node = torch.tanh(.7*self.node+.3*sensory+.5*messages)
        self.previous = sensory.clone()
        return F.interpolate(self.node[None], scale_factor=4,
                             mode='nearest')[0].clone()


class LiquidReservoir:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.input = _random((8, 6, 1, 1), 47, .2)
        self.recurrent = _random((8, 8, 3, 3), 53, .015)
        self.recurrent *= (_random((8, 8, 3, 3), 59) > .03).float()
        self.readout = torch.zeros((6, 8, 1, 1))
        self.reset_state()

    def learned_tensors(self):
        return (self.readout,)

    def reset_state(self):
        self.state = torch.zeros((8, self.height, self.width))
        self.mean = torch.zeros((8, 1, 1))

    @torch.no_grad()
    def step(self, value, *, learn=False):
        drive = F.conv2d(value[None], self.input)[0]
        drive += F.conv2d(self.state[None], self.recurrent, padding=1)[0]
        self.state = torch.tanh(.8*self.state+drive-self.mean)
        self.mean.lerp_(self.state.mean((1, 2), keepdim=True), .01)
        reconstruction = F.conv2d(self.state[None], self.readout)[0]
        if learn:
            error = value-reconstruction
            update = error.flatten(1) @ self.state.flatten(1).T
            exposure = self.state.square().sum((1, 2)).clamp(min=1.)
            self.readout.add_(.03*(update/exposure[None]).reshape_as(
                self.readout)).clamp_(-1., 1.)
        return torch.cat((self.state, reconstruction)).clone()


FAMILIES = dict(slow_feature=SlowFeature, common_fate=CommonFate,
                sparse_reconstruction=SparseReconstruction,
                attractor=AttractorAssembly, synchrony=SynchronyBinding,
                relational_graph=RelationalGraph, reservoir=LiquidReservoir)
