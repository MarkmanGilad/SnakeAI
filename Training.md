# Training Overview

## State Representation

The game board is a `17×17` grid converted to a PyTorch tensor.

| Cell Value | Meaning |
|---|---|
| `0` | Empty cell |
| `1` | Snake head |
| `2` | Snake body |
| `3` | Mouse (food) |

**Tensor shape:** `(1, 1, 17, 17)` — (batch, channels, height, width).

Bombs are **not** encoded in the state tensor.

**Snake init:** Head placed randomly at `(row, col)` where both ∈ `[4, 12]`, with one body segment in a random cardinal direction. Starting length = 2.

**Mouse init:** Placed randomly at `(row, col)` where both ∈ `[1, 15]`, ensuring no overlap with the snake.

---

## Action Space

| Constant | Value | Direction |
|---|---|---|
| `ACTION_UP` | 1 | Row − 1 |
| `ACTION_DOWN` | 2 | Row + 1 |
| `ACTION_LEFT` | 3 | Col − 1 |
| `ACTION_RIGHT` | 4 | Col + 1 |

4 discrete actions total.

### Epsilon-Greedy Selection

Linear decay: ε = ε_start − (ε_start − ε_final) × epoch / decay for `epoch < decay`, then ε = ε_final.

| Parameter | Value |
|---|---|
| ε start | 1.0 |
| ε final | 0.05 |
| Decay | 400 epochs |

If `random() < ε` → random action; otherwise → `argmax(Q(state))`.

---

## Reward Structure

| Event | Reward | Constant |
|---|---|---|
| Eating (mouse consumed) | +5.0 | `REWARD_EAT` |
| Winning (board full) | +5.0 | `REWARD_WIN` |
| Losing (wall/self collision or bomb) | −3.0 | `REWARD_LOSE` |
| Moving closer to mouse | +0.5 | `REWARD_CLOSER` |
| Moving farther from mouse | −0.1 | `REWARD_FARTHER` |

### Closer / Farther Logic

The action direction is compared against the relative position of the mouse to the snake head. Only one axis is checked per action (row for UP/DOWN, column for LEFT/RIGHT). If the action moves the head toward the mouse on the relevant axis → `REWARD_CLOSER`; otherwise → `REWARD_FARTHER`.

### Step Order

1. Compute `closer` reward as baseline.
2. Check board-full → `(REWARD_WIN, done=True)`.
3. Check if action leads to eating → override reward with `REWARD_EAT`, grow snake, spawn new mouse.
4. Otherwise, execute move → if collision → `(REWARD_LOSE, done=True)`.
5. Bomb logic (active when `score >= 10`): random spawn with probability 1.5%, 3-second timer, Chebyshev distance ≤ 1 kills snake → `(REWARD_LOSE, done=True)`.
6. Return `(reward, done=False)`.

---

## DQN Architecture

```
Conv2d(1 → 16, 3×3, stride=1, pad=1) → ReLU
Conv2d(16 → 16, 3×3, stride=1, pad=1) → ReLU
Flatten → Linear(4624 → 64) → ReLU
Linear(64 → 4)
```

---

## Training Loop (Double DQN)

### Two Networks

- **Online network** (`player`) — used for action selection and Q-value computation.
- **Target network** (`player_hat`) — initialized as a deep copy of the online network.

### Per-Epoch Loop

1. Reset environment (snake, mouse, score).
2. **Collect experience:** each step:
   - Select action via epsilon-greedy.
   - Execute action → get `(reward, done)`.
   - Store `(state, action, reward, next_state, done)` in replay buffer.
   - If `done` → break.
3. **Train** (every step, if buffer size ≥ 1,000):
   - Sample 128 transitions from buffer.
   - Compute `Q(s, a)` from the online network.
   - **DDQN action selection:** online network selects best next action.
   - **DDQN evaluation:** target network evaluates that action's Q-value.
   - Compute loss, backprop, optimizer step.

### Target Network Update

Hard copy every 20 epochs via `load_state_dict`.

A soft update method (Polyak averaging, τ = 0.001) exists but is not used in the current trainer.

---

## Loss Function

$$L = \text{MSE}\Big(Q(s, a),\; r + \gamma \cdot Q_{\hat{\theta}}\big(s',\, \arg\max_{a'} Q_\theta(s', a')\big) \cdot (1 - \text{done})\Big)$$

When `done = 1`, the target collapses to just the reward.

The **Double DQN** decoupling (online network selects action, target network evaluates it) reduces Q-value overestimation bias.

---

## Hyperparameter Summary

| Parameter | Value |
|---|---|
| Board size | 17 × 17 |
| State tensor shape | (1, 1, 17, 17) |
| Actions | 4 |
| γ (discount factor) | 0.99 |
| ε schedule | Linear 1.0 → 0.05 over 400 epochs |
| Batch size | 128 |
| Buffer capacity | 500,000 |
| Min buffer before training | 1,000 |
| Target update | Hard copy every 20 epochs |
| Learning rate | 0.001 (halved at epochs 5k, 10k, 20k) |
| Loss | MSE with DDQN targets |
