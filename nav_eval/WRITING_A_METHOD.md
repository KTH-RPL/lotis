# Evaluating your own method

To benchmark your own navigation method against LoTIS, implement the `NavigationMethod`
interface, register it, and point the runner at it. You do not need to touch the runner,
the dataset loader, or the metrics.

## The interface

Subclass `nav_eval.NavigationMethod` and implement three methods:

```python
from typing import Any, Dict

from nav_eval import Action, NavigationMethod, register_method


class MyMethod(NavigationMethod):
    def setup(self) -> None:
        """Load models / allocate resources once, before any episode runs."""

    def reset_episode(self, episode: Dict[str, Any]) -> None:
        """Prepare for a new trajectory-query episode (e.g. encode the reference)."""

    def act(self, observation: Dict[str, Any]) -> Action:
        """Return a velocity command for the current observation."""


register_method("my_method", MyMethod)
```

A few optional hooks are also available:

- `set_simulator(self, simulator)` — called once after `setup()`. Use it if your method
  needs the Habitat pathfinder (e.g. `simulator.sim.pathfinder`, `simulator.compute_path`)
  for local planning or obstacle avoidance. Most methods can ignore it.
- `visualize(self, observation)` — called each step when the run is started with
  `--visualize` (see below). No-op by default.
- `cleanup(self)` — called once after the run to release resources.

`__init__(self, config)` receives the parsed contents of your method config YAML (see
below) as a plain dict; the base class stores it on `self.config`.

## What `act` returns

```python
Action(linear_velocity: float, angular_velocity: float)
```

- `linear_velocity` — forward speed in m/s (agent-local, +forward).
- `angular_velocity` — yaw rate in rad/s (+left).

The runner clamps these to the simulator limits (`max_linear_velocity`,
`max_angular_velocity` from the dataset config — defaults 0.5 m/s and 1.2 rad/s) and
integrates them for one control step (`1 / control_hz`, default 0.1 s). There is no
explicit STOP action: an episode ends when the agent reaches the goal
(within `success_distance_threshold`, default 0.5 m) or hits `max_steps` (timeout).

## The observation passed to `act`

| Key             | Type                    | Description                                             |
|-----------------|-------------------------|---------------------------------------------------------|
| `rgb`           | `np.ndarray (H, W, 3)`  | Current RGB view, `uint8`.                              |
| `rgb_pil`       | `PIL.Image`             | Same frame as a PIL image.                              |
| `intrinsics`    | `np.ndarray (3, 3)`     | Pinhole camera matrix for the current frame.            |
| `position`      | `np.ndarray (3,)`       | Agent world position (Habitat Y-up, `-Z` forward).      |
| `rotation`      | `quaternion`            | Agent world orientation.                                |
| `step`          | `int`                   | Step index within the episode.                          |

## The episode passed to `reset_episode`

| Key              | Type                     | Description                                                  |
|------------------|--------------------------|-------------------------------------------------------------|
| `rgb_images`     | `list[PIL.Image]`        | The reference trajectory frames, in order.                  |
| `tracking_mode`  | `str`                    | `"forward"` (goal = last frame), `"backward"` (goal = first frame), or `"goal_point"`. |
| `goal_point_id`  | `int` or `None`          | Index of the goal frame in `rgb_images` for `goal_point` mode. |
| `scene_id`, `trajectory_id`, `query_id` | `str`     | Episode identifiers.                                        |

## A minimal example

A trivial baseline that drives straight ahead. Save it as `my_method.py` anywhere
importable:

```python
from typing import Any, Dict

from nav_eval import Action, NavigationMethod, register_method


class GoStraight(NavigationMethod):
    def setup(self) -> None:
        pass

    def reset_episode(self, episode: Dict[str, Any]) -> None:
        pass

    def act(self, observation: Dict[str, Any]) -> Action:
        return Action(linear_velocity=0.4, angular_velocity=0.0)


register_method("go_straight", GoStraight)
```

Run it:

```bash
uv run python -m nav_eval.run \
    --dataset gibson \
    --method-module my_method \
    --method go_straight \
    --max-episodes 8
```

`--method-module` is imported before the method is looked up, which triggers the
`register_method(...)` call at import time. Use `--method-config` semantics by adding a
`method` block to a custom `--config` YAML if your method needs parameters; the config
dict is delivered to your `__init__`.

## Visualization

Run with `--visualize` to stream the evaluation to a [Rerun](https://rerun.io) viewer.
The runner logs the live camera frame at entity path `"camera"` and the goal image at
`"goal"` every step. After each `act`, it calls your method's `visualize(observation)`,
where you can log anything you want.

Log overlays under `"camera/..."` so they compose with the live view. For example, LoTIS
logs its predicted image-space trajectory points as `rr.Points2D` under `"camera/points"`
and the closest reference frame as a separate image — see `visualize` in
`nav_eval/methods/lotis.py`. `rerun` is already a dependency; import it inside `visualize`:

```python
def visualize(self, observation):
    import rerun as rr
    # ... compute pixel positions from your prediction and the frame size ...
    rr.log("camera/points", rr.Points2D(positions=positions, colors=colors))
```

## Output

Each case writes to `eval_results/<dataset>/<case_name>/<method>/`:

- `results.json` — per-episode metrics plus an aggregate summary.
- `worker.log` — the full verbose run log (setup, per-episode, timing).
- `eval_config.yaml`, `method_config.yaml` — the exact configs used.

Compare two result directories (e.g. your method vs LoTIS):

```bash
uv run python -m nav_eval.compare \
    --reference eval_results/gibson/off_start_forward_cross_camera/lotis \
    --candidate eval_results/gibson/off_start_forward_cross_camera/my_method
```
