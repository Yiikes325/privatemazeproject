import os
import sys
import time
import datetime
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import PReLU



#5x5 matrix
maze_matrix = np.array([
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1.],
    [0., 1., 1., 1., 1.],
    [1., 0., 0., 1., 0.],
    [0., 1., 0., 1., 1.]
])

#10x10 matrix
#maze_matrix = np.array([
    #[1., 1., 1., 1., 1., 0., 0., 1., 1., 0.],
    #[1., 1., 1., 0., 1., 1., 0., 1., 0., 0.],
    #[0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #[1., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
    #[0., 1., 0., 1., 1., 1., 0., 1., 1., 1.],
    #[1., 0., 1., 1., 1., 0., 0., 1., 1., 0.],
    #[1., 1., 1., 0., 1., 1., 1., 1., 0., 0.],
    #[1., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
    #[1., 0., 1., 1., 1., 0., 1., 0., 1., 0.],
    #[0., 1., 0., 1., 1., 1., 1., 1., 1., 1.]
#])

#25x25 matrix
#maze_matrix = np.array([
#    [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1.],
#    [1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1.],
#    [0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.],
#    [1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.],
#    [0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.],
#    [1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0.],
#    [1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#    [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0.],
#    [1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#    [0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0.],
#    [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1.],
#    [1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1.],
#    [0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.],
#    [1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.],
#    [0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.],
#    [1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0.],
#    [1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
#    [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0.],
#    [1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#    [0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0.],
#    [0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0.],
#    [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1.],
#    [1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1.],
#    [0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.],
#    [1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.]
#])

#Command to show the maze on matplotlib:
#qLearningMaze = qLearningMaze(maze_matrix)
#show(qLearningMaze)

trainingAgentMark = 0.5 #Matplotlib purposes
#Integers related to the agent's movements.
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actionDictionary = { #Action dictionary containing the number of actions the agent can perform.
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

numberOfActions = len(actionDictionary)
epsilon = 0.2 #Frequency level that calculates how much exploration is performed.

class QLearningMaze(object):
    def __init__(self, maze_matrix, agent=(0,0)): #Program is initialised, agent is placed at cell 0,0.
        self._maze = np.array(maze_matrix) #Creation of the maze.
        nrows, ncols = self._maze.shape #Number of rows and columns formulate the shape of the maze.
        self.target = (nrows-1, ncols-1) #The target is the cell on the bottom right, where the agent must navigate to.
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0] #Calculated the number of free cells available in the rows and columns.
        self.free_cells.remove(self.target) #Removes the target cell
        if self._maze[self.target] == 0.0: #If the target cell is a wall...
            raise Exception("Invalid maze. The cell on the bottom right must be available to navigate to.") #Raise an exception stating that the target cell must be a path.
        if not agent in self.free_cells: #If the agent is currently starting on a wall...
            raise Exception("Invalid location. The agent must start at a free cell in order to navigate through the maze.") #Raise an exception stating that the agent must start on a free cell.
        self.reset(agent)

    def reset(self, agent): #Function to reset the maze, allowing for the agent to start a new instance.
        self.agent = agent
        self.maze_matrix = np.copy(self._maze)
        nrows, ncols = self.maze_matrix.shape
        row, col = agent
        self.maze_matrix[row, col] = trainingAgentMark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze_matrix.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action): #Updates the current state of the maze.
        nrows, ncols = self.maze_matrix.shape
        nrow, ncol, nmode = agent_row, agent_col, mode = self.state

        if self.maze_matrix[agent_row, agent_col] > 0.0:
            self.visited.add((agent_row, agent_col))#If the agent is no longer the starting position and is moving, mark the current cell as visited and add to the list.

        valid_actions = self.valid_actions() #Checks to see the current valid actions available to the agent.
                
        if not valid_actions: #If none of the actions are valid...
            nmode = 'blocked' #Inform the agent that the current action is blocked.
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:
            mode = 'invalid'

        #Creates a new state.
        self.state = (nrow, ncol, nmode)

    def get_reward(self): #Reward system for the agent.
        agent_row, agent_col, mode = self.state
        nrows, ncols = self.maze_matrix.shape
        if agent_row == nrows-1 and agent_col == ncols-1: #If the agent makes it to the target cell...
            return 1.0 #Award full marks
        if mode == 'blocked': #If the agent attempts to enter a blocked spot...
            return self.min_reward - 1 #Give the agent the minimum reward, minus 1
        if (agent_row, agent_col) in self.visited: #If the agent returns to a visited spot...
            return -0.25 #Remove 0.25 points from the agent.
        if mode == 'invalid': #if the agent attempts to enter an invalid spot...
            return -0.75 #Remove 0.75 points from the agent.
        if mode == 'valid': #If the agent goes to a valid spot...
            return -0.04 #Remove 0.04 points so that the agent doesn't try to take centuries to get to the correct spot, and provide it with motivation.

    def act(self, action): #The result of the action the agent would take.
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        environmentState = self.observe()
        return environmentState, reward, status

    def observe(self): #The current state of the environment.
        canvas = self.draw_environment()
        environmentState = canvas.reshape((1, -1))
        return environmentState

    def draw_environment(self): #Draws the maze based on the matrix provided.
        canvas = np.copy(self.maze_matrix)
        nrows, ncols = self.maze_matrix.shape
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        row, col, valid = self.state
        canvas[row, col] = trainingAgentMark
        return canvas


    def game_status(self): #The current status of the game. Checks to see if the game has concluded.
        if self.total_reward < self.min_reward:
            return 'lose'
        trainingAgentRow, trainingAgentColumn, mode = self.state
        nrows, ncols = self.maze_matrix.shape
        if trainingAgentRow == nrows-1 and trainingAgentColumn == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze_matrix.shape
        if row == 0: #If the location ABOVE the agent is 0...
            actions.remove(1) #Up is no longer an option
        elif row == nrows-1: #If the location BELOW the agent is 0...
            actions.remove(3) #Down is no longer an option

        if col == 0: #If the location TO THE LEFT of the agent is 0...
            actions.remove(0) #Left is no longer an option
        elif col == ncols-1: #If the location TO THE RIGHT of the agent is 0...
            actions.remove(2) #Right is no longer an option

        if row>0 and self.maze_matrix[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze_matrix[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze_matrix[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze_matrix[row,col+1] == 0.0:
            actions.remove(2)

        return actions

#Function to show the maze on matplotlib. Can be done by defining the maze matrix, then typing show(qLearningMaze). Not necessarily needed.
#def show(qLearningMaze):
    #plt.grid('on')
    #nrows, ncols = qLearningMaze.maze_matrix.shape
    #ax = plt.gca()
    #ax.set_xticks(np.arange(0.5, nrows, 1))
    #ax.set_yticks(np.arange(0.5, ncols, 1))
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #canvas = np.copy(qLearningMaze.maze_matrix)
    #for row,col in qLearningMaze.visited:
    #    canvas[row,col] = 0.6
    #trainingAgentRow, trainingAgentColumn, _ = qLearningMaze.state
    #canvas[trainingAgentRow, trainingAgentColumn] = 0.3   
    #canvas[nrows-1, ncols-1] = 0.9 
    #img = plt.imshow(canvas, interpolation='none', cmap='gray')
    #return img

def play_game(trainingModel, qLearningMaze, trainingAgentCell): #Plays the maze game.
    qLearningMaze.reset(trainingAgentCell)
    environmentState = qLearningMaze.observe()
    while True:
        prev_environmentState = environmentState
        #Gets the next action
        q = trainingModel.predict(prev_environmentState, verbose = None)
        action = np.argmax(q[0])

        #Applies the current action, obtains reward, goes to the next state.
        environmentState, reward, gameStatus = qLearningMaze.act(action)
        if gameStatus == 'win':
            return True
        elif gameStatus == 'lose':
            return False
        
class Experience(object): #Memorises all the locations the agent has gone through.
    def __init__(self, trainingModel, max_memory=100, discount=0.95):
        self.trainingModel = trainingModel
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.numberOfActions = trainingModel.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, environmentState):
        return self.trainingModel.predict(environmentState, verbose = None)[0]

    def get_data(self, data_size=10):
        environment_size = self.memory[0][0].shape[1]   #State of the 1D size.
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, environment_size))
        targets = np.zeros((data_size, self.numberOfActions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            environmentState, action, reward, environmentState_next, game_over = self.memory[j]
            inputs[i] = environmentState
            # There should be no target values for actions not taken.
            targets[i] = self.predict(environmentState)
            #Q(s, a) formula, explained in Chapter 3.
            Q_sa = np.max(self.predict(environmentState_next))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
    
def qtrain(trainingModel, maze_matrix, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 1000) #Number of epochs
    max_memory = opt.get('max_memory', 1000) #How much memory kept in regards to the experience class above.
    data_size = opt.get('data_size', 50) #Highlights the amount of samples used in each epoch.
    weights_file = opt.get('weights_file', "") #Method that SHOULD hopefully allow to read a file containing trained agent, and continue to use it to look at its performance.
    name = opt.get('name', 'trainingModel')
    start_time = datetime.datetime.now()

    #Allows for previously trained mazes to be retrained for testing purposes.
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        trainingModel.load_weights(weights_file)

    #Constructs environment/game from numpy array.
    qLearningMaze = QLearningMaze(maze_matrix)

    #Initializes the experience replay object.
    experience = Experience(trainingModel, max_memory=max_memory)

    win_history = []   #Table for the win and loss history.
    n_free_cells = len(qLearningMaze.free_cells) #Number of free cells.
    hsize = qLearningMaze.maze_matrix.size//2   #Size of the window history.
    win_rate = 0.0 #Win rate is set to 0.

    for epoch in range(n_epoch):
        loss = 0.0
        agent_cell = random.choice(qLearningMaze.free_cells)
        qLearningMaze.reset(agent_cell)
        game_over = False

        #Obtain the initial environment state, based on the 1d array.
        environmentState = qLearningMaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qLearningMaze.valid_actions()
            if not valid_actions: break
            prev_environmentState = environmentState
            #Obtain the next action in accordance to our policy.
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_environmentState))

            #Apply the action taken, give the neural network the reward and update the game's status.
            environmentState, reward, game_status = qLearningMaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            #Stores the experiences the neural network has gone through.
            episode = [prev_environmentState, action, reward, environmentState, game_over]
            experience.remember(episode)
            n_episodes += 1

            #Trains the neural network model.
            inputs, targets = experience.get_data(data_size=data_size)
            h = trainingModel.fit(inputs, targets, epochs=8, batch_size=16, verbose=None,)
            loss = trainingModel.evaluate(inputs, targets, verbose=None)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize #Calculates the win rate.
    
        dt = datetime.datetime.now() - start_time
        t = calculatedTime(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        if win_rate > 0.9 : epsilon = 0.05 #Program checks to see if the agent has managed to navigate through all the free cells and the route is optimal.
        if sum(win_history[-hsize:]) == hsize: #If it is...
            print("Reached 100%% win rate at epoch: %d" % (epoch,)) #It will state the epoch at which it had achieved a perfect win rate.
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    trainingModel.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(trainingModel.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = calculatedTime(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

#Method used to make time strings more readable. Assists in calculating how long it took for the agent to reach 100% win rate.
def calculatedTime(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

#Constructing the model. Uses 2 hidden layers, with both inputs being the size of the matrix. The output layer consists of the number of actions performed.
#The loss function is measured using a means standard error.    
def buildTrainingModel(maze_matrix, lr=0.001):
    trainingModel = Sequential()
    trainingModel.add(Dense(maze_matrix.size, input_shape=(maze_matrix.size,)))
    trainingModel.add(PReLU())
    trainingModel.add(Dense(maze_matrix.size))
    trainingModel.add(PReLU())
    trainingModel.add(Dense(numberOfActions))
    trainingModel.compile(optimizer='adam', loss='mse')
    return trainingModel

#Command to create the model.
trainingModel = buildTrainingModel(maze_matrix)
qtrain(trainingModel, maze_matrix, epochs=1000, max_memory=8 * maze_matrix.size, data_size=32)