import pygame
import torch
from Environment import *
from AgentDQN import *
from ReplayBuffer import ReplayBuffer
from Constant import *
import wandb

def main():
    num = 103

    pygame.init()
    env = Environment()

    best_score = 0

    ######## PARAMS ########

    player = AgentDQN()
    player_hat = AgentDQN()
    player_hat.DQN = player.DQN.copy()

    buffer = ReplayBuffer()

    loss = torch.tensor(0.0)
    avg = 0
    scores, losses, avg_score = [], [], []

    optim = torch.optim.Adam(player.DQN.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, SCHEDULER_MILESTONES, gamma=SCHEDULER_GAMMA
    )

    
    project = "SnakeAI"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        id = f'{project}_{num}',
        resume="never",
        # track hyperparameters and run metadata
        config={
        "name": f"SnakeAI",
        "learning_rate": LEARNING_RATE,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": EPOCHS,
        "start_epoch": START_EPOCH,
        "epsilon_start": EPSILON_START,
        "epsilon_final": EPSILON_FINAL,
        "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "target_update_freq": TARGET_UPDATE_FREQ,
        "min_buffer_size": MIN_BUFFER_SIZE,
        "buffer_capacity": BUFFER_CAPACITY,
        "max_steps_without_eat": MAX_STEPS_WITHOUT_EAT,
        "Model":str(player.DQN),
        "REWARD_WIN":REWARD_WIN,
        "REWARD_LOSE": REWARD_LOSE,
        "REWARD_CLOSER": REWARD_CLOSER,
        "REWARD_FARTHER":REWARD_FARTHER,
        "REWARD_EAT": REWARD_EAT
        })

    for epoch in range(START_EPOCH, EPOCHS):

        env.reset()
        state = env.to_tensor()
        step = 0
        steps_since_eat = 0
        prev_score = 0
        while True:

            print(step, end="\r")
            step += 1
            steps_since_eat += 1

            pygame.event.get()
            env.graphics.draw(env)
            ######## SAMPLE ENVIRONMENT ########
			
            action = player.get_action(env, epoch)

            reward, done = env.move_env(action)
            next_state = env.to_tensor()
            
            buffer.push(
                state,
                torch.tensor([[action]], dtype=torch.int64),
                torch.tensor([[reward]], dtype=torch.float32),
                next_state,
                torch.tensor([[done]], dtype=torch.float32),
            )

            if done:
                best_score = max(best_score, env.score)
                break

            if env.score > prev_score:
                steps_since_eat = 0
                prev_score = env.score

            if steps_since_eat >= MAX_STEPS_WITHOUT_EAT:
                break

            state = next_state


            if len(buffer) < MIN_BUFFER_SIZE:
                continue

            ######## TRAIN ########

            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            Q_values = player.Q(states, actions)

            next_actions, _ = player.get_Actions_Values(next_states)
            with torch.no_grad():
                Q_hat_values = player_hat.Q(next_states, next_actions)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_values, dones)

            loss.backward()
            optim.step()
            optim.zero_grad()
            

        ######## TARGET UPDATE ########
        scheduler.step()
        if epoch % TARGET_UPDATE_FREQ == 0:
            player_hat.fix_update(player.DQN)


        print(
            f"num: {num} epoch: {epoch} loss: {loss:.6f} LR: {scheduler.get_last_lr()} "
            f"score: {env.score} best_score: {best_score} step: {step}"
        )
        
        
        wandb.log({
			"loss": loss.item(),
			"score": env.score,
			"best_score": best_score,
            "step": step
		})


        if epoch % LOG_INTERVAL == 0:
            scores.append(env.score)
            losses.append(loss.item())

        avg = (avg * (epoch % LOG_INTERVAL) + env.score) / (epoch % LOG_INTERVAL + 1)

        if (epoch + 1) % LOG_INTERVAL == 0:
            avg_score.append(avg)
            print(f"average score last {LOG_INTERVAL} games: {avg}")
            avg = 0


if __name__ == "__main__":
    main()
