import retro

# Create the enviroment
env = retro.make('SuperMarioBros-Nes', 'Level1-1')
env.reset()

# We need to loop while not DONE
done = False
while not done:
    # See whats happenin
    env.render()

    # Call a random button press from the controller
    # action = env.action_space.sample()
    
    action = [0,0,1,0,0,0,0,1,1,1,0,0]
    #print(action)

    # ob = image of the screen at the time of the action
    # reward = amount of reward that he earns from the scenario.json file
    # done = if the done 
    # info = data.json

    ob, reward, done, info = env.step(action) # Send vars to the enviroment
    print(reward)

