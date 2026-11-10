"""Fitting-only optimizations for the first-frame pen pilot."""
import torch

from unionmimic.leap_sharpa.model import X
from unionmimic.leap_sharpa.pen_collect import PenReplay
from unionmimic.leap_sharpa.pen_model import PenModel


class QuickPenModel(PenModel):
    def physical_transition(self, s, command, c, *, scale, external_force):
        reference = self.tasks.target(*self.labels(c), self.phase(s))
        target, residual = self.controller(reference['q'], command, s[:, X['action']])
        core, _ = self.physics(s[:, :79], target, self.physics_context(s, c, external_force))
        # The fitting losses read physical state and reference phase. Sensor
        # reconstruction is separately supervised on measured single steps.
        return torch.cat((core, s[:, 79:145], external_force, residual, s[:, X['last_contact']],
                          torch.zeros_like(s[:, X['stall']]), s[:, X['progress']] + 1, s[:, 177:]), -1)


class QuickPenReplay(PenReplay):
    terminal_rows = None

    def add(self, **row):
        self.terminal_rows = None
        super().add(**row)

    def failure_rows(self, batch):
        if self.terminal_rows is None:
            self.terminal_rows = (self.done == 1).nonzero()
        terminal = self.terminal_rows
        n_terminal = min(len(terminal), batch // 4)
        want = batch - n_terminal
        rows, envs, count = [], [], 0
        for _ in range(30):
            r = torch.randint(min(self.G, self.L), (2 * batch,), device=self.device)
            e = torch.randint(self.N, (2 * batch,), device=self.device)
            valid = self.epi[r, e] >= 0
            rows.append(r[valid])
            envs.append(e[valid])
            count += int(valid.sum())
            if count >= want:
                break
        if count < want:
            raise RuntimeError('Not enough valid transitions for failure sampling')
        r, e = torch.cat(rows)[:want], torch.cat(envs)[:want]
        if n_terminal:
            selected = terminal[torch.randint(len(terminal), (n_terminal,), device=self.device)]
            r, e = torch.cat((r, selected[:, 0])), torch.cat((e, selected[:, 1]))
        return r, e
