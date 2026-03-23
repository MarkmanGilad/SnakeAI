import pygame
import torch
from Graphics import *
from Environment import Environment
from AgentDQN import AgentDQN
from ReplayBuffer import ReplayBuffer
from Environment import *
import os
import wandb

def main ():
    pygame.init()
    # params and models
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    env = Environment()
    player = AgentDQN(devive=device)
    player_hat = AgentDQN(devive=device)
    player_hat.DQN = player.DQN.copy()
    batch_size = 128
    buffer = ReplayBuffer(path=None)
    learning_rate = 0.0001
    epochs = 200000
    start_epoch = 0
    C, tau = 3, 0.001
    loss = torch.tensor(0)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[5000*1000, 10000*1000, 15000*1000, 20000*1000, 25000*1000, 30000*1000], gamma=0.5)
    step = 0

    # checkpoint Load
    num = 400
    checkpoint_path = f"Data/checkpoint{num}.pth"
    buffer_path = f"Data/buffer{num}.pth"

    # Wandb.init
    wandb.init(
        # set the wandb project where this run will be logged
        project="SnakeAI",
        id=f'SnakeAI {num}',
        # track hyperparameters and run metadata
        config={
        "name": f"SnakeAI DDQN {num}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": epochs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C,
        "tau":tau,
        "Model":str(player.DQN),
        "device": str(device)
        }
    )

    for epoch in range(start_epoch, epochs):
        
        # Episode loop - one game loop
        env.restart()
        end_of_game = False
        state = env.state()
        while not end_of_game:
            print (step, end='\r')
            step += 1
            
            # Play and Sample Environement
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
            
            action = player.get_Action(state=state, epoch=epoch)
            reward, done = env.move(action=action)
            next_state = env.state()
            buffer.push(state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state, torch.tensor(done, dtype=torch.float32))
            if done:
                best_score = max(best_score, state.score)
                break

            state = next_state

            print("epoch: " + str (epoch))
            pygame.display.update()
            # clock.tick(FPS)
            
            if len(buffer) < MIN_BUFFER:
                continue
            
            # Train
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.Q(states, actions)
            next_actions, _ = player.get_Actions_Values(next_states)
            Q_hat_Values = player_hat.Q(next_states, next_actions)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

        # Update target network
        if epoch % C == 0:
            player_hat.fix_update(dqn=player.DQN)
        
        # Printing and saving
        print (f'epoch: {epoch} loss: {loss:.7f} LR: {scheduler.get_last_lr()} step: {step} ' \
            f'score: {state.score} best_score: {best_score}')
        step = 0
        if epoch % 10 == 0:
            scores.append(state.score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + state.score) / (epoch % 10 + 1)
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            wandb.log ({
                "score": state.score,
                "loss": loss.item(),
                "avg_score": avg
            })
            print (f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores':scores,
                'avg_score': avg_score
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)

if __name__ == "__main__":
    main ()