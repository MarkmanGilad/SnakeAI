# Update Log

## Session — April 5, 2026

### Created `Constant.py`
Centralized all magic numbers and parameters into a single `Constant.py` file. All other files use `from Constant import *`.

**Constants added:**

| Category | Constants |
|---|---|
| Game | `BOARD_SIZE`, `SQUARE_SIZE`, `FPS` |
| Actions | `ACTION_UP`, `ACTION_DOWN`, `ACTION_LEFT`, `ACTION_RIGHT`, `ACTIONS` |
| Rewards | `REWARD_WIN`, `REWARD_LOSE`, `REWARD_CLOSER`, `REWARD_FARTHER`, `REWARD_EAT` |
| DQN Architecture | `CONV1_IN/OUT_CHANNELS`, `CONV2_IN/OUT_CHANNELS`, `CONV_KERNEL_SIZE`, `CONV_STRIDE`, `CONV_PADDING`, `FC_INPUT_SIZE`, `FC_HIDDEN_SIZE`, `OUTPUT_SIZE`, `GAMMA` |
| Epsilon Greedy | `EPSILON_START`, `EPSILON_FINAL`, `EPSILON_DECAY` |
| Training | `MIN_BUFFER_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `START_EPOCH`, `TARGET_UPDATE_FREQ`, `SCHEDULER_MILESTONES`, `SCHEDULER_GAMMA`, `LOG_INTERVAL` |
| Replay Buffer | `BUFFER_CAPACITY` |
| Soft Update | `TAU` |
| Bomb | `BOMB_MIN_DISTANCE`, `BOMB_TIMER_SECONDS`, `BOMB_COOLDOWN`, `BOMB_SPAWN_PROBABILITY`, `SECOND_SCREEN_SCORE` |
| Snake/Mouse Init | `SNAKE_INIT_MIN`, `SNAKE_INIT_MAX`, `MOUSE_INIT_MIN`, `MOUSE_INIT_MAX` |
| Colors | `COLOR_BLUE`, `COLOR_LIGHTBLUE`, `COLOR_DARK_BROWN`, `COLOR_LIGHT_BROWN`, `COLOR_PINK`, `COLOR_LIGHT_PINK`, `COLOR_WHITE`, `COLOR_BOMB_AREA`, `COLOR_BOMB_OUTER`, `COLOR_BOMB_INNER`, `COLOR_EXIT_BUTTON`, `COLOR_EXIT_BUTTON_HOVER` |
| Fonts | `FONT_PATH`, `FONT_SIZE`, `FONT_SIZE_SMALL`, `FONT_SIZE_LARGE` |
| UI | `OVERLAY_OPACITY`, `BUTTON_WIDTH`, `BUTTON_HEIGHT`, `BUTTON_BORDER_RADIUS`, `BUTTON_BORDER_WIDTH`, `SCORE_POSITION` |

### Updated Files

- **DQN.py** — Removed local `layer1`, `layer2`, `output_size`, `gamma`. Replaced conv/linear layer params and `gamma` with constants.
- **AgentDQN.py** — Removed `epsilon_start`, `epsilon_final`, `epsiln_decay`. Replaced `[1,2,3,4]` with `ACTIONS`, `tau` defaults with `TAU`, epsilon params with `EPSILON_*`.
- **Environment.py** — Removed `REWARD_WIN/LOSE/CLOSER/FARTHER/EAT`. Replaced action numbers (`1/2/3/4`) with `ACTION_UP/DOWN/LEFT/RIGHT`, board sizes with `BOARD_SIZE`, bomb params with `BOMB_*`, init ranges with `SNAKE_INIT_*` / `MOUSE_INIT_*`, screen threshold with `SECOND_SCREEN_SCORE`.
- **Graphics.py** — Removed `BOARD_SIZE`, `SQUARE_SIZE`, and color constants (`BLUE`, `LIGHTBLUE`, `DARK_BROWN`, `LIGHT_BROWN`, `PINK`, `LIGHT_PINK`). Replaced all font paths/sizes, colors, button dimensions, overlay opacity, and score position with constants.
- **ReplayBuffer.py** — Removed `capacity = 500000`. Uses `BUFFER_CAPACITY`.
- **Trainer.py** — Removed `MIN_BUFFER = 1000` and local training variables (`batch_size`, `learning_rate`, `epochs`, `start_epoch`, `C`). Uses constants directly.
- **HumanAgent.py** — Replaced action return values (`1/2/3/4`) with `ACTION_UP/DOWN/LEFT/RIGHT`.
- **Environment_Human.py** — Replaced `clock.tick(10)` with `clock.tick(FPS)`.

### Bug Fixes

- **DQN.py** — Reverted `.detach()` from `loss()` — no longer needed since the graph is never built for target values.
- **Trainer.py** — Wrapped `player_hat.Q()` call in `torch.no_grad()` instead of detaching after the fact. This avoids building the computation graph for the target network entirely, rather than building it and discarding it. Removed dead variable `end_of_game`.

### Reward Tuning

Rebalanced rewards to make eating the dominant signal and death a strong penalty. The old `REWARD_CLOSER` (+0.5) was too high — the snake could farm proximity rewards nearly equal to eating. Added `MAX_STEPS` to prevent infinite looping episodes.

| Constant | Old | New | Reason |
|---|---|---|---|
| `REWARD_EAT` | +5 | +10 | Make eating the dominant reward signal |
| `REWARD_WIN` | +5 | +10 | Match eat reward scale |
| `REWARD_LOSE` | −3 | −10 | Death must outweigh accumulated proximity rewards |
| `REWARD_CLOSER` | +0.5 | +0.1 | Gentle directional hint, not a farmable reward |
| `REWARD_FARTHER` | −0.1 | −0.2 | Slightly stronger discouragement for moving away |
| `MAX_STEPS` | ∞ | 1000 | New — prevents infinite safe-looping episodes |

**Files changed:**
- **Constant.py** — Updated reward values, added `MAX_STEPS = 1000`.
- **Trainer.py** — Added `MAX_STEPS` check to break the episode loop.

### Deeper Network Architecture

The old network had only a 5×5 receptive field on a 17×17 board (2 conv layers with 16 filters each) and an extreme FC bottleneck (4624→64). The snake couldn't see enough of the board to plan paths or avoid trapping itself.

| Layer | Old | New |
|---|---|---|
| Conv1 | 1→16 | 1→32 |
| Conv2 | 16→16 | 32→64 |
| Conv3 | — | 64→64 (new) |
| Receptive field | 5×5 | 7×7 |
| FC hidden | 64 (4624→64, 72:1) | 256 (18496→256, 72:1) |

**Files changed:**
- **Constant.py** — Updated conv channel sizes, added `CONV3_*` constants, increased `FC_HIDDEN_SIZE` to 256.
- **DQN.py** — Added 3rd conv layer + ReLU in `__init__` and `forward`.

### Episode Step Limit

Replaced fixed `MAX_STEPS` with `MAX_STEPS_WITHOUT_EAT = 500`. The step counter now resets every time the snake eats, so a snake that keeps eating can play indefinitely and learn to navigate as a long snake. If it goes 500 steps without eating, the episode ends.

**Files changed:**
- **Constant.py** — Replaced `MAX_STEPS = 1000` with `MAX_STEPS_WITHOUT_EAT = 500`.
- **Trainer.py** — Added `steps_since_eat` counter that resets on score increase, breaks episode when limit reached.

### wandb Fix

Added `resume="never"` to `wandb.init()` to prevent silently resuming old runs when reusing a run ID.

**Files changed:**
- **Trainer.py** — Added `resume="never"` parameter.

### Removed Unused Soft Update

`soft_update` method and `TAU` constant were never used — only `fix_update` (hard copy) is called in the Trainer.

**Removed:**
- **Constant.py** — Deleted `TAU = 0.001`.
- **AgentDQN.py** — Deleted `soft_update` method, removed unused `tau` param from `fix_update`.

### Epsilon Final Reduction

Reduced `EPSILON_FINAL` from 0.05 to 0.01. At 0.05, the agent made a random move every ~20 steps — too likely to kill a long snake. At 0.01, it's every ~100 steps.

**Files changed:**
- **Constant.py** — `EPSILON_FINAL`: 0.05 → 0.01.

### wandb Config Cleanup

Added missing parameters to wandb config: `EPSILON_START`, `EPSILON_FINAL`, `MIN_BUFFER_SIZE`, `BUFFER_CAPACITY`, `MAX_STEPS_WITHOUT_EAT`. Renamed `decay` → `epsilon_decay`, `C` → `target_update_freq` for clarity.

**Files changed:**
- **Trainer.py** — Updated `wandb.init` config with all important hyperparameters.

### Checkpoint Saving

The Trainer was not saving model parameters — all progress was lost on stop/crash. Added two save mechanisms:
- **Best model** — saves to `Data/best_{num}.pth` when score ties or beats the best (threshold ≥ 5 to skip trivial early saves).
- **Periodic** — saves to `Data/checkpoint_{num}_epoch{N}.pth` every `CHECKPOINT_INTERVAL = 1000` epochs.
- Moved `best_score` update to after checkpoint check so new bests are properly detected.

**Files changed:**
- **Constant.py** — Added `CHECKPOINT_INTERVAL = 1000`, `CHECKPOINT_DIR = "Data"`.
- **Trainer.py** — Added `import os`, checkpoint save logic, reordered `best_score` update.
