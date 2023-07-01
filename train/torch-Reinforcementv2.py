import pickle
from random import random, randint

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.random.seed(2018)


# from torch import sigmoid


class DiskEnv(gym.Env):
    def __init__(self):
        self.n_disk = 10000
        self.rate = 0.01
        self.redundancy_strategy = (14, 10)
        self.disk_scrubbing_freq = 7
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_disk,), dtype=np.int)
        self.action_space = gym.spaces.Discrete(9)
        self.default_MTTDL = self.calculate_MTTDL(self.rate, 14, 10, 1 / (7 * 24))
        self.default_space = 1
        self.default_MTTD = 3.5
        self.default_cost = self.n_disk * (1 / 7)


    def step(self, action):
        if action == 0:
            pass
        elif action < 5:
            # Modify redundancy strategy
            # if action == 3:
            #     self.redundancy_strategy = (14, 10)
            if action == 1:
                if self.redundancy_strategy[0] > 8:
                    self.redundancy_strategy = (self.redundancy_strategy[0] - 1, self.redundancy_strategy[1] - 1)
            elif action == 2:
                if self.redundancy_strategy[0] < 24:
                    self.redundancy_strategy = (self.redundancy_strategy[0] + 1, self.redundancy_strategy[1] + 1)
            if action == 3:
                # If the current frequency is greater than the minimum value, decrement it by one
                if self.disk_scrubbing_freq > 1:
                    self.disk_scrubbing_freq -= 1
            elif action == 4:
                # If the current frequency is less than the maximum value, increment it by one
                if self.disk_scrubbing_freq < 30:
                    self.disk_scrubbing_freq += 1
            self.repair_state()
        else:
            if action == 5:
                if self.redundancy_strategy[0] > 9:
                    self.redundancy_strategy = (self.redundancy_strategy[0] - 2, self.redundancy_strategy[1] - 2)
            elif action == 6:
                if self.redundancy_strategy[0] < 23:
                    self.redundancy_strategy = (self.redundancy_strategy[0] + 2, self.redundancy_strategy[1] + 2)
            # Modify disk scrubbing frequency
            if action == 7:
                # If the current frequency is greater than the minimum value, decrement it by one
                if self.disk_scrubbing_freq > 2:
                    self.disk_scrubbing_freq -= 2
            elif action == 8:
                # If the current frequency is less than the maximum value, increment it by one
                if self.disk_scrubbing_freq < 29:
                    self.disk_scrubbing_freq += 2

        # Update state
        num_failures = 0
        next_state = torch.tensor(self.state, dtype=torch.float32)
        for i in range(self.n_disk):
            if next_state[i] == -1:
                continue
            if np.random.rand() < self.rate:
                next_state[i] = 0
                num_failures += 1

        n1, k1 = self.redundancy_strategy[0], self.redundancy_strategy[1]
        info = {
            'MTTDL': (self.calculate_MTTDL(self.rate, n1, k1, 1 / (
                        self.disk_scrubbing_freq * 24))) / self.default_MTTDL,
            'space': self.calculate_space(n1, k1),
            'MTTD': self.calculate_MTTD() / self.default_MTTD,
            'cost': self.calculate_cost() / self.default_cost,
            'redundancy_strategy': self.redundancy_strategy,
            'disk_scrubbing_freq': self.disk_scrubbing_freq
        }

        # Calculate reward
        reward = self.calculate_reward(
            info['MTTDL'],
            info['space'],
            info['MTTD'],
            info['cost'],
            num_failures)


        # Check if done
        done = torch.any(next_state == 0)

        self.state = next_state.numpy()

        return self.state, reward, done, info

    # Define a function to calculate combinations
    # def n_choose_k(n, k):
    #     return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    def check_state(self):
        data_disk_num, parity_disk_num = self.redundancy_strategy
        n_data = self.n_disk // (data_disk_num + parity_disk_num) * data_disk_num  # Number of data disks
        n_parity = self.n_disk // (data_disk_num + parity_disk_num) * parity_disk_num  # Number of parity disks
        n_check = self.n_disk // (data_disk_num + parity_disk_num)  # Number of check disks per group

        for i in range(n_data):
            disk_idx = i * (data_disk_num + parity_disk_num)
            if self.state[disk_idx] == -1:
                continue
            if np.sum(self.state[disk_idx:disk_idx + data_disk_num + parity_disk_num]) != n_data + n_parity:
                # If the total number of data disks and parity disks in this group does not equal to the total number of data disks and parity disks, it means this group does not meet the redundancy strategy and needs repair
                self.repair_state()
                return

    def calculate_reward(self, MTTDL_diff, Space_diff, MTTD_diff, Cost_diff, num_failures):
        default_fail = self.n_disk * (0.03)
        MTTDL_factor = 10 * num_failures / default_fail
        Space_factor = 15
        MTTD_factor = 10
        cost_factor = 10

        reward = math.exp(MTTDL_factor * math.log(abs(MTTDL_diff) + 1e-10))
        reward *= math.exp(Space_factor * math.log(abs(Space_diff) + 1e-10))
        reward *= math.exp(MTTD_factor * math.log(abs(MTTD_diff) + 1e-10))
        reward *= math.exp(cost_factor * math.log(abs(Cost_diff) + 1e-10))

        return reward

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Define a function to calculate MTTDL
    def calculate_MTTDL(self, rate, n, k, mu):
        # afr: annual failure rate
        # n: total number of disks
        # k: minimum number of available disks
        # mu: disk repair rate
        # Calculate the failure rate per hour
        lam = rate / 8760
        n_choose_k = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        # Calculate the numerator part in the MTTDL formula
        numerator = (1 / lam) * (1 - n_choose_k * lam ** (n - k) * mu ** k)
        # Calculate the denominator part in the MTTDL formula
        denominator = n * lam
        # Return the MTTDL value
        return numerator / denominator

    def calculate_space(self, n1, k1):
        # n1: number of encoded blocks for code 1
        # k1: number of data blocks for code 1
        # n2: number of encoded blocks for code 2
        # k2: number of data blocks for code 2
        return (14 / 10) / (n1 / k1) - 1

    def calculate_MTTD(self):
        # Calculate the current MTTD
        # print(self.disk_scrubbing_freq)
        return self.disk_scrubbing_freq / 2

    def calculate_cost(self):
        # Calculate the cost of disk scrubbing
        scrubbing_cost = self.n_disk * (1 / self.disk_scrubbing_freq)
        return scrubbing_cost

    def reset(self):
        self.state = np.ones(self.n_disk)
        return self.state

    def render(self):
        # Visualize the environment state, such as printing the current disk state and redundancy strategy
        print(f"Current disk state: {self.state}")
        print(f"Current redundancy strategy: {self.redundancy_strategy}")
        print(f"Current disk scrubbing frequency: {self.disk_scrubbing_freq}")

    def repair_state(self):
        """Repair the disk state"""
        n_data, n_parity = self.redundancy_strategy
        self.state = np.ones(self.n_disk)
        # if n_data > self.n_disk:
        #     raise ValueError("Too many data disks")
        # if n_data + n_parity > self.n_disk:
        #     raise ValueError("Too many disks in total")
        # if self.n_disk % (n_data + n_parity) != 0:
        #     raise ValueError("Number of disks must be divisible by number of data and parity disks")

        n_block = self.n_disk // (n_data + n_parity)
        for i in range(n_parity):
            parity = 0
            for j in range(n_block):
                disk = i + j * n_parity + n_data
                if disk >= self.n_disk:
                    raise ValueError("Disk index out of range")
                parity ^= int(self.state[disk])
            self.state[i + n_data::n_parity] = parity


def computeDay(group):
    group = group.sort_values('date')  # ordino in base ai giorni... dal più recente al meno
    group['DayToFailure'] = list(range(group.shape[0] - 1, -1, -1))
    return group


def divideInLevel(x):
    if x.Label == 0:
        return 'Good'  # Good
    elif x.DayToFailure <= 9:
        return 'Alert'  # Alert
    elif x.DayToFailure <= 21:
        return 'Warning '  # Warning
    else:
        return 'Very Fair'


def tolerance_acc(x):
    if x.pred == 'c_Good':
        return x.vero == 'c_Good' or x.vero == 'c_Very Fair'

    if x.pred == 'c_Very Fair':
        return x.vero == 'c_Good' or x.vero == 'c_Very Fair' or x.vero == 'c_Warning'

    if x.pred == 'c_Warning':
        return x.vero == 'c_Very Fair' or x.vero == 'c_Warning' or x.vero == 'c_Alert'

    if x.pred == 'c_Alert':
        return x.vero == 'c_Warning' or x.vero == 'c_Alert'


def binary_classification_pred(x):
    if x.pred == 'c_Good' or x.pred == 'c_Very Fair':
        return 0
    else:
        return 1


def binary_classification_label(x):
    if x.vero == 'c_Good' or x.vero == 'c_Very Fair':
        return 0
    else:
        return 1


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    cols, names = list(), list()
    dataclass = data[data.columns[-1:]]
    data = data.drop(columns=['serial_number', 'Class'], axis=1)
    columns = data.columns
    # input sequence (t-n, ... t-1)  #non arrivo all'osservazione corrente
    for i in range(n_in - 1, 0, -1):
        cols.append(data.shift(i))
        names += [(element + '(t-%d)' % (i)) for element in columns]

    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [(element + '(t)') for element in columns]
        else:
            names += [(element + '(t+%d)' % (i)) for element in columns]

    cols.append(dataclass)  # appendo le ultime cinque colonne
    names += ['Class']

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def balancing_by_replication(X_train):
    alert = X_train[X_train.c_Alert == 1]
    vfair = X_train[X_train['c_Very Fair'] == 1]
    warn = X_train[X_train.c_Warning == 1]
    # 'c_Alert','c_Good','c_Very Fair','c_Warning'
    good = X_train[X_train.c_Good == 1]  # sono i buoni

    size_good = good.shape[0]

    while alert.shape[0] < size_good:
        app = alert.sample(min(alert.shape[0], size_good - alert.shape[0]), replace=False)
        alert = alert.append(app)

    while vfair.shape[0] < size_good:
        app = vfair.sample(min(vfair.shape[0], size_good - vfair.shape[0]), replace=False)
        vfair = vfair.append(app)

    while warn.shape[0] < size_good:
        app = warn.sample(min(warn.shape[0], size_good - warn.shape[0]), replace=False)
        warn = warn.append(app)

    good = good.append(alert)
    good = good.append(vfair)
    good = good.append(warn)
    return good


def data():

    listLabels = ['c_Alert', 'c_Good', 'c_Very Fair', 'c_Warning']
    finestra = 14
    # 15model\Train-WDC-WD30EFRX BalckDaUsare

    df = pd.read_csv('15model\Train-WDC-WD30EFRX.csv', sep=',')
    df = df.dropna()
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d').dt.date
    df = df.drop(['CurrentPendingSectorCount', 'ReallocatedSectorsCount', 'model', 'capacity_bytes'], axis=1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[['ReportedUncorrectableErrors', 'HighFlyWrites', 'TemperatureCelsius',
        'RawCurrentPendingSectorCount', 'RawReadErrorRate', 'SpinUpTime',
        'RawReallocatedSectorsCount', 'SeekErrorRate', 'PowerOnHours']] = scaler.fit_transform(
        df[['ReportedUncorrectableErrors',
            'HighFlyWrites', 'TemperatureCelsius',
            'RawCurrentPendingSectorCount',
            'RawReadErrorRate', 'SpinUpTime',
            'RawReallocatedSectorsCount',
            'SeekErrorRate', 'PowerOnHours']])

    dfHour = df.groupby(['serial_number']).apply(computeDay)
    dfHour = dfHour[dfHour.DayToFailure <= 45]
    dfHour = dfHour.drop(columns=['date'])
    dfHour['Class'] = dfHour.apply(divideInLevel, axis=1)
    dfHour = dfHour.drop(columns=['Label', 'DayToFailure', 'serial_number'], axis=1)
    dfHour = dfHour.reset_index()
    dfHour = dfHour.drop(columns=['level_1'], axis=1)

    # creo le sequenze
    print('Creazione Sequenze')
    dfHourSequence = dfHour.groupby(['serial_number']).apply(series_to_supervised, n_in=finestra, n_out=1, dropnan=True)
    dfHourSequence = pd.concat([dfHourSequence, pd.get_dummies(dfHourSequence.Class, prefix='c')], axis=1).drop(
        ['Class'], axis=1)
    numberClasses = len(listLabels)

    # divisione in train validation e split
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
                                                        dfHourSequence[dfHourSequence.columns[-numberClasses:]],
                                                        test_size=0.25,
                                                        random_state=42)

    # Split training set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.25,  # 0.25 x 0.8 = 0.2
                                                      random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(dfHourSequence[dfHourSequence.columns[:-numberClasses]],
    #                                                   dfHourSequence[dfHourSequence.columns[-numberClasses:]] ,
    # #                                                   test_size=0.2, random_state=42)
    # print(y_train[0].sum(),y_train[1].sum(),y_train[3].sum(),y_train[4].sum())
    N = [y_train[col].sum() for col in y_train.columns]
    total_sum = sum(N)
    ratios = [num / total_sum for num in N]
    W = [1000,1,500,500]
    score = weighted_sum = sum(ratios[i] * W[i] for i in range(len(ratios)))
    print(y_train.columns)
    print(score)
    if score<16:
        return 1
    elif score<25:
        return 3
    elif score<40:
        return 6
    else :
        return 10
    del dfHourSequence
    del dfHour


class Agent(object):
    def __init__(self, env):
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 500
        self.steps_done = 0
        self.env = env
        self.lr = 0.01
        self.gamma = 0.99
        self.epsilon = 0.1
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def choose_action(self, state):
        self.steps_done += 1
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay)

        if torch.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32)
            action_values = self.model(state)
            action = torch.argmax(action_values).item()
        return action


    # def learn(self, state, action, reward, next_state, done):
    #     state = torch.tensor(state, dtype=torch.float32)
    #     action_values = self.model(state)
    #
    #     target = reward
    #     print('target :' , target)
    #     if not done:
    #         next_state = torch.tensor(next_state, dtype=torch.float32)
    #         next_action_values = self.model(next_state)
    #         target += self.gamma * torch.max(next_action_values).item()
    #
    #     target_values = action_values.clone().detach()
    #     print('target_values:' ,target_values)
    #     target_values[0][action] = target
    #
    #     loss = self.criterion(action_values, target_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    def learn(self, state, action, reward, next_state, done):
        # self.optimizer.zero_grad()

        state = torch.tensor(state, dtype=torch.float32)
        action_values = self.model(state)

        target = reward
        # print('target :', target)
        if not done:
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_action_values = self.model(next_state)
            target += self.gamma * torch.max(next_action_values).item()

        target_values = action_values.clone().detach()
        # print('target_values:', target_values)
        if len(target_values.shape) != 0:
            # target_values[0][action] = target
            target_values[action]= target

            loss = self.criterion(action_values, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            print("target_values' dimension is 0, skip the optimization step.")

def main():
    H = data()
    # Define empty lists
    mttdl_list = []
    space_list = []
    mttd_list = []
    cost_list = []
    redundancy_strategy_list = []
    disk_scrubbing_freq_list = []
    reward_list = []

    env = DiskEnv()  # Create environment object
    env.rate = 0.1*H
    agent = Agent(env) # Create agent object

    n_episode = 1000  # Define the number of episodes for training
    for i in range(n_episode):  # Loop over each episode
        state = env.reset()  # Reset the environment state
        done = False  # Initialize the termination flag as False

        while not done:  # Loop until termination
            action = agent.choose_action(state)  # Agent chooses an action based on the current state
            next_state, reward, done, info = env.step(action)  # Environment updates the state based on the chosen action and returns the next state, reward, and termination flag

            # Append data to the lists at the end of each round
            mttdl_list.append(info['MTTDL'])
            space_list.append(info['space'])
            mttd_list.append(info['MTTD'])
            cost_list.append(info['cost'])
            redundancy_strategy_list.append(info['redundancy_strategy'])
            disk_scrubbing_freq_list.append(info['disk_scrubbing_freq'])
            reward_list.append(reward)

            agent.learn(state, action, reward, next_state, done)  # Agent updates model parameters based on experience

            state = next_state  # Update the current state with the next state

            if done:  # If termination occurs, print some information and break the loop
                break



    a = randint(1, 1000)
    file = '001'
    filename = f"{a}_{file}.pkl"
    # Save the three lists to the same file
    with open('save/filename', 'wb') as f:
        pickle.dump((mttdl_list,
            space_list,
            mttd_list,
            cost_list,
            redundancy_strategy_list,
            disk_scrubbing_freq_list,
            reward_list.append), f)

    env.close()  # Close the environment

if __name__ == "__main__":
    main()
