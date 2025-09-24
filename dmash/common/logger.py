# Adapted from https://github.com/danijar/elements

import collections
import concurrent.futures
import json
import re
import time
import pathlib

import numpy as np
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, initial_step, outputs, multiplier=1):
        assert outputs, 'Provide a list of logger outputs.'
        self.step = initial_step
        self.outputs = outputs
        self.multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, step, prefix=None):
        step = int(step)
        self.step = step
        step *= self.multiplier
        for name, value in dict(mapping).items():
            name = f'{prefix}/{name}' if prefix else name
            if isinstance(value, str):
                pass
            elif isinstance(value, plt.Figure):
                pass
            else:
                value = np.asarray(value)
                if len(value.shape) not in (0, 1, 2, 3, 4):
                    raise ValueError(
                        f"Shape {value.shape} for name '{name}' cannot be "
                        "interpreted as scalar, vector, image, or video.")
            self._metrics.append((step, name, value))

    def write(self, fps=False):
        if fps:
            value = self._compute_fps()
            if value is not None:
                self.add({'fps': value}, step=self.step)
        if not self._metrics:
            return
        for output in self.outputs:
            output(tuple(self._metrics))
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self.step) * self.multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return None
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class AsyncOutput:
    def __init__(self, callback, parallel=True):
        self._callback = callback
        self._parallel = parallel
        if parallel:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._future = None

    def __call__(self, summaries):
        if self._parallel:
            self._future and self._future.result()
            self._future = self._executor.submit(self._callback, summaries)
        else:
            self._callback(summaries)


class TerminalOutput:
    def __init__(self, pattern=r'.*', name=None):
        self._pattern = re.compile(pattern)
        self._name = name
        try:
            import rich.console
            self._console = rich.console.Console()
        except ImportError:
            self._console = None

    def __call__(self, summaries):
        step = max(s for s, _, _, in summaries)
        scalars = {
            k: float(v) for _, k, v in summaries
            if isinstance(v, np.ndarray) and len(v.shape) == 0}
        scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        if self._console:
            if self._name:
                self._console.rule(f'[green bold]{self._name} (Step {step})')
            else:
                self._console.rule(f'[green bold]Step {step}')
            self._console.print(' [blue]/[/blue] '.join(
                f'{k} {v}' for k, v in formatted.items()))
            print('')
        else:
            message = ' / '.join(f'{k} {v}' for k, v in formatted.items())
            message = f'[{step}] {message}'
            if self._name:
                message = f'[{self._name}] {message}'
            print(message, flush=True)

    def _format_value(self, value):
        value = float(value)
        if value == 0:
            return '0'
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            value = value.rstrip('0')
            value = value.rstrip('0')
            value = value.rstrip('.')
            return value
        else:
            value = f'{value:.1e}'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '')
            value = value.replace('+', '')
            value = value.replace('-0', '-')
        return value


class JSONLOutput(AsyncOutput):
    def __init__(
            self, filename, pattern=r'.*',
            strings=False, parallel=True, config=None):
        super().__init__(self._write, parallel)
        self._filename = pathlib.Path(filename)
        self._filename.parent.mkdir(parents=True, exist_ok=True)
        self._pattern = re.compile(pattern)
        self._strings = strings
        if config is not None:
            with open(self._filename.parent / "metadata.json", "w") as f:
                json.dump(config, f)

    def _write(self, summaries):
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str) and self._strings:
                bystep[step][name] = value
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                bystep[step][name] = float(value)
            if isinstance(value, np.ndarray) and len(value.shape) == 1:
                bystep[step][name] = value.tolist()
        lines = ''.join([
            json.dumps({'step': step, **values}) + '\n'
            for step, values in bystep.items()])
        with open(self._filename, "a") as f:
            f.write(lines)


class WandBOutput:
    def __init__(self, pattern=r'.*', **kwargs):
        import wandb
        self._pattern = re.compile(pattern)
        wandb.init(**kwargs)

    def __call__(self, summaries):
        import wandb
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str):
                bystep[step][name] = value
            elif isinstance(value, plt.Figure):
                bystep[step][name] = wandb.Image(value)
                plt.close(value)
            elif len(value.shape) == 0:
                bystep[step][name] = float(value)
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)
            elif len(value.shape) in (2, 3):
                value = value[..., None] if len(value.shape) == 2 else value
                assert value.shape[3] in [1, 3, 4], value.shape
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                assert value.shape[3] in [1, 3, 4], value.shape
                value = np.transpose(value, [0, 3, 1, 2])
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                bystep[step][name] = wandb.Video(value)
        for step, metrics in bystep.items():
            wandb.log(metrics, step=step)

