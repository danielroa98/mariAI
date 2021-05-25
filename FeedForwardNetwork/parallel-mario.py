import retro
import numpy as np
import cv2
import neat
import gym
#import gym_pull
import pickle

# (Research this !)
#gym_pull.pull('github.com/ppaquette/gym-super-mario')

# Im going to limit the actions of Mario, only allowing him to move to the RIGHT and JUMP
ACTIONS = [
    [0 ,0, 0, 0, 0, 0, 0, 1, 0], # Right
    [0 ,0, 0, 0, 0, 0, 0, 1, 1], # Right Jump
    [0 ,0, 0, 0, 0, 0, 0, 0, 1], # Jump
    [1 ,0, 0, 0, 0, 0, 0, 1, 0], # Dash Right
    [0 ,0, 0, 0, 0, 0, 0, 1, 1], # Dash Right Jump
]

Action_Desc = [
    "Right",
    "Right Jump",
    "Jump",
    "Dash Jump",
    "Dash Right Jump"
]

# Class of my worker (thread)
class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
    
    # Normal function form test2
    def work(self):
        # Create the enviroment
        self.env = retro.make('SuperMarioBros-Nes', 'Level1-1')
        ob = self.env.reset() # Get the very first image from the Level 1

        # Variables for our inputs
        ob,_,_,_ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)


        # Network (Going to try FeedForwardNetwork)
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

         # Track fitness -> output from the network, how successful the genome was (reward function)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0 # Track X position
        xpos_max = 0 # Track if mario is at the end of the level
        done = False
        imgarray = []

        cv2.namedWindow("NN", cv2.WINDOW_NORMAL)
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        while not done:

            #self.env.render()
            cv2.imshow('main', ob)
            # Downsize the screenshot for our neural network (using opencv)
            ob = cv2.resize(ob, (inx,iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)# Make the screenshot Grayscale, fewer inputs = faster solutions
            
            
            cv2.imshow('NN', ob)
            cv2.waitKey(1)

            ob = np.reshape(ob, (inx,iny)) # Reshape to fit the NN

            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0,254), (-1,+1))
            

            actions = net.activate(imgarray)# Output of 12 values from our NN
            ind = actions.index(max(actions))

            #print(actions)
            #print(Action_Desc[ind])

            # Record the result of inputing the nnOutput into our emulator, the reward from that and either if its done or not.
            ob, reward, done, info = self.env.step(actions) # Send vars to the enviroment

            # Make Mario done if he doesnt achieve a certain "goal" in a certain amount of frames
            #xpos = info['xscrollLo']

            # If Mario reaches the end of the level, supposedely, the fitness would be around 3253, in that case, done.
            if fitness_current > 3252 or info['lives']<2:
                done = True

            # If Mario goes further to the right that he has ever been, he gets 1 point
            #if xpos > xpos_max:
            #    fitness_current += 1
            #    xpos_max = xpos

            fitness_current += reward

            # Mario reset a counter everytime he hits a new best fitness
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # Use counter to determine done 
            # Basically, this genome has 250 attempts to move to the right each pixel.
            if done or counter == 250:
                done = True
                #print(self.genome.genome_id, fitness_current)

            #imgarray.clear()

        print(fitness_current)
        return fitness_current


def eval_genomes(genome, config):
    worker = Worker(genome,config)
    return worker.work()

# NEAT Config file
config = neat.Config(neat.DefaultGenome, # activation functions, 0.05 mutation
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     'config-test1') #name of the configuration file

# Population Configuration
#p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-492')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) # Every 10 generations, save Checkpoint

pe = neat.ParallelEvaluator(1, eval_genomes)

winner = p.run(pe.evaluate)

# Save the winner with PICKLE!
with open('winner.pickle', 'wb') as output:
    pickle.dump(winner, output,1)