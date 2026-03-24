import pygame
import torch
from Environment import *
from AgentDQN import *
from ReplayBuffer import ReplayBuffer
import wandb

MIN_BUFFER = 1000

def main():
    num = 68

    pygame.init()
    env = Environment()

    best_score = 0

    ######## PARAMS ########

    player = AgentDQN()
    player_hat = AgentDQN()
    player_hat.DQN = player.DQN.copy()

    batch_size = 128
    buffer = ReplayBuffer()

    learning_rate = 0.001
    epochs = 200000
    start_epoch = 0
    C = 20

    loss = torch.tensor(0.0)
    avg = 0
    scores, losses, avg_score = [], [], []

    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, [5000, 10000, 20000], gamma=0.5
    )

    
    project = "SnakeAI"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        id = f'{project}_{num}',
        # track hyperparameters and run metadata
        config={
        "name": f"SnakeAI",
        "learning_rate": learning_rate,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": epochs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": gamma,
        "batch_size": batch_size, 
        "C": C,
        "Model":str(player.DQN),
        "REWARD_WIN":REWARD_WIN,
        "REWARD_LOSE": REWARD_LOSE,
        "REWARD_CLOSER": REWARD_CLOSER,
        "REWARD_FARTHER":REWARD_FARTHER,
        "REWARD_EAT": REWARD_EAT
        #"device": str(device)
        })

    for epoch in range(start_epoch, epochs):

        env.reset()
        end_of_game = False
        state = env.to_tensor()
        step = 0
        while not end_of_game:

            print(step, end="\r")
            step += 1

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

            state = next_state


            if len(buffer) < MIN_BUFFER:
                continue

            ######## TRAIN ########

            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            Q_values = player.Q(states, actions)

            next_actions, _ = player.get_Actions_Values(next_states)
            Q_hat_values = player_hat.Q(next_states, next_actions)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_values, dones)

            loss.backward()
            optim.step()
            optim.zero_grad()
            

        ######## TARGET UPDATE ########
        scheduler.step()
        if epoch % C == 0:
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


        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + env.score) / (epoch % 10 + 1)

        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            print(f"average score last 10 games: {avg}")
            avg = 0


if __name__ == "__main__":
    main()
