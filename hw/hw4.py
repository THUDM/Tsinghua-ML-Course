import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, learning_rate, gamma):
        super(ActorCritic, self).__init__()
        self.data = []
        self.gamma = gamma

        self.fc1 = nn.Linear(ob_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, ac_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        # *****START OF YOUR CODE*****
        prob = None
        # *****END OF YOUR CODE*****
        return prob

    def v(self, x):
        # *****START OF YOUR CODE*****
        v = None
        # *****END OF YOUR CODE*****
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_lst, dtype=torch.float),
        )
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        # *****START OF YOUR CODE*****
        loss = None
        # *****END OF YOUR CODE*****

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    # Hyperparameters


def main():
    # *****START OF YOUR CODE*****
    env = gym.make("CartPole-v1")
    n_rollout = 300
    print_interval = 20
    max_episodes = 10000
    model = ActorCritic(
        ob_dim=env.observation_space.shape[0],
        ac_dim=env.action_space.n,
        hidden_dim=256,
        learning_rate=0.0002,
        gamma=0.98,
    )
    # *****END OF YOUR CODE*****

    score = 0.0
    for n_epi in range(max_episodes):
        done = False
        s = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float().unsqueeze(0)).squeeze()
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break
                
                # Comment out these two lines if running on a server
                if n_epi % print_interval == 0:
                    env.render()

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score / print_interval
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
