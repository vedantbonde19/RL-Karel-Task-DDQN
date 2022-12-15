import gym
import numpy as np
from gym import spaces
import json

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, jsondata, render_mode = None):

        if (jsondata != None):
            with open(jsondata) as f:
                data = json.load(f)

        else:
            with open("C:/Users/vedan/Downloads/RL_Project/0_task.json") as f:
                data = json.load(f)

        self.gridsz_num_rows = data["gridsz_num_rows"]
        self.gridsz_num_cols = data["gridsz_num_cols"]
        self.pregrid_agent_row = data["pregrid_agent_row"]
        self.pregrid_agent_col = data["pregrid_agent_col"]
        self.pregrid_agent_dir = data["pregrid_agent_dir"]
        self.postgrid_agent_row = data["postgrid_agent_row"]
        self.postgrid_agent_col = data["postgrid_agent_col"]
        self.postgrid_agent_dir = data["postgrid_agent_dir"]
        self.walls = np.array(data["walls"])
        self.pregrid_markers = np.array(data["pregrid_markers"])
        self.postgrid_markers = np.array(data["postgrid_markers"])
        self.agent_row = data["pregrid_agent_row"]
        self.agent_col = data["pregrid_agent_col"]
        self.grid_size = (self.gridsz_num_rows, self.gridsz_num_cols)

        self.agent_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        self.agent_grid[self.pregrid_agent_row][self.pregrid_agent_col] = 1

        self.agent_dir = -1
        self.agent_dir_arr = np.zeros(4)

        self.postgrid_dir = -1
        self.postgrid_dir_arr = np.zeros(4)

        if(self.pregrid_agent_dir == "east"):
            self.agent_dir = 0
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "south"):
            self.agent_dir = 1
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "west"):
            self.agent_dir = 2
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "north"):
            self.agent_dir = 3
            self.agent_dir_arr[self.agent_dir] = 1

        if(self.postgrid_agent_dir == "east"):
            self.postgrid_dir = 0
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "south"):
            self.postgrid_dir = 1
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "west"):
            self.postgrid_dir = 2
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "north"):
            self.postgrid_dir = 3
            self.postgrid_dir_arr[self.postgrid_dir] = 1


        self.walls_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for wall in self.walls:
            self.walls_grid[wall[0]][wall[1]] = 1

        self.markers_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for marker in self.pregrid_markers:
            self.markers_grid[marker[0]][marker[1]] = 1

        self.agent_postgrid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        self.agent_postgrid[self.postgrid_agent_row][self.postgrid_agent_col] = 1

        self.markers_postgrid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for marker in self.postgrid_markers:
            self.markers_postgrid[marker[0]][marker[1]] = 1


        self.action_space = spaces.Discrete(6)
        #self.observation_space = spaces.Dict(
        #   {
        #        "agent_loc": spaces.MultiBinary(self.grid_size), 
        #        "agent_dir": spaces.Discrete(4), 
        #        "agent_marker": spaces.MultiBinary(self.grid_size), 
        #        "postgrid_loc": spaces.MultiBinary(self.grid_size), 
        #        "postgrid_dir":spaces.Discrete(4), 
        #        "postgrid_marker": spaces.MultiBinary(self.grid_size), 
        #        "walls": spaces.MultiBinary(self.grid_size)
        #    }
        #)

        tot_size = 16 + 4 + 16 + 16 + 4 + 16 + 16

        self.observation_space = spaces.MultiBinary(tot_size)


    
    def _get_obs(self):

        obs = {
            "agent_loc": self.agent_grid,
            "agent_dir": self.agent_dir_arr,
            "agent_marker": self.markers_grid,
            "postgrid_loc": self.agent_postgrid,
            "postgrid_dir": self.postgrid_dir_arr,
            "postgrid_marker": self.markers_postgrid,
            "walls": self.walls_grid
        }

        return obs

    

    def _get_info(self):

        info = {
            "agent_loc": self.agent_grid,
            "agent_dir": self.agent_dir,
            "agent_marker": self.markers_grid,
            "postgrid_location": self.agent_postgrid,
            "postgrid_dir": self.postgrid_dir,
            "postgrid_marker": self.markers_postgrid,
            "walls": self.walls_grid
        }

        return info

    def flatten_obs(self, obs):
        agent_loc = obs["agent_loc"].flatten()
        agent_dir = obs["agent_dir"].flatten()
        agent_marker = obs["agent_marker"].flatten()
        postgrid_loc = obs["postgrid_loc"].flatten()
        postgrid_dir = obs["postgrid_dir"].flatten()
        postgrid_marker = obs["postgrid_marker"].flatten()
        walls = obs["walls"].flatten()
        ar = np.concatenate((agent_loc, agent_dir, agent_marker, postgrid_loc, postgrid_dir, postgrid_marker, walls))
        return ar

    def step(self, action): ## Define step function

        y = self.agent_col
        x = self.agent_row

        reward = -1
        done = False

        agent_d = self.agent_dir

        if action == 0: ## Move 

            if agent_d == 0:
                y = y + 1 

                if (y > 3):
                    reward = -10
                    done = True # Crash into wall 


                elif (self.walls_grid[x][y] == 1):
                    reward = -10
                    done = True # Crash into wall 

                else: ## Transition move

                    self.agent_grid[self.agent_row][self.agent_col] = 0
                    self.agent_grid[x][y] = 1
                    self.agent_col = y

            elif agent_d == 1:
                x = x + 1

                if (x > 3):
                    reward = -10
                    done = True # Crash into wall 

                elif (self.walls_grid[x][y] == 1):
                    reward = -10
                    done = True # Crash into wall 
                    
                else: ## Transition move

                    self.agent_grid[self.agent_row][self.agent_col] = 0
                    self.agent_grid[x][y] = 1
                    self.agent_row = x

            if agent_d == 2:
                y = y - 1 

                if (y < 0):
                    reward = -10
                    done = True # Crash into wall 

                elif (self.walls_grid[x][y] == 1):
                    reward = -10
                    done = True # Crash into wall 

                else: ## Transition move

                    self.agent_grid[self.agent_row][self.agent_col] = 0
                    self.agent_grid[x][y] = 1
                    self.agent_col = y


            if agent_d == 3:
                x = x - 1 

                if (x < 0):
                    reward = -10
                    done = True # Crash into wall 

                elif (self.walls_grid[x][y] == 1):
                    reward = -10
                    done = True # Crash into wall 

                else: ## Transition move

                    self.agent_grid[self.agent_row][self.agent_col] = 0
                    self.agent_grid[x][y] = 1
                    self.agent_row = x


        elif action == 1:
            if self.agent_dir == 0:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 3
                self.agent_dir_arr[self.agent_dir] = 1


            elif self.agent_dir == 1:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 0
                self.agent_dir_arr[self.agent_dir] = 1

            elif self.agent_dir == 2:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 1
                self.agent_dir_arr[self.agent_dir] = 1


            elif self.agent_dir == 3:
                self.agent_dir_arr[self.agent_dir] = 0
                self. agent_dir = 2
                self.agent_dir_arr[self.agent_dir] = 1

        elif action == 2:
            if self.agent_dir == 0:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 1
                self.agent_dir_arr[self.agent_dir] = 1


            elif self.agent_dir == 1:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 2
                self.agent_dir_arr[self.agent_dir] = 1

            elif self.agent_dir == 2:
                self.agent_dir_arr[self.agent_dir] = 0
                self.agent_dir = 3
                self.agent_dir_arr[self.agent_dir] = 1

            elif self.agent_dir == 3:
                self.agent_dir_arr[self.agent_dir] = 0
                self. agent_dir = 0
                self.agent_dir_arr[self.agent_dir] = 1


        elif action == 3: ## Pick Marker

            if self.markers_grid[self.agent_row][self.agent_col] == 1:

                if self.markers_postgrid[self.agent_row][self.agent_col] == 0:
                    reward = 5 #for normal model
                else:
                    reward = -10 #for normal model
                
                self.markers_grid[self.agent_row][self.agent_col] = 0

            else:

                reward = -10 #for normal model 
                done = True ## Crash if no marker is placed.

        elif action == 4: ## Drop Marker

            if self.markers_grid[self.agent_row][self.agent_col] == 0:

                if self.markers_postgrid[self.agent_row][self.agent_col] == 1:
                    reward = 5 #in case of normal model
                else:
                    reward = -10 #for normal model

                self.markers_grid[self.agent_row][self.agent_col] = 1

            else:

                reward = -10 # for normal model
                done = True
                self.markers_grid[self.agent_row][self.agent_col] = 1

        
        elif action == 5: ## Finish 

            if ((np.array_equal(self.agent_grid, self.agent_postgrid) and np.equal(self.agent_dir, self.postgrid_dir) and np.array_equal(self.markers_grid, self.markers_postgrid))):

                reward = 100
                done = True
            else:
                reward = -10
                done = True 

        
        observation = self._get_obs()
        observation = self.flatten_obs(observation)
        info = self._get_info()

        return observation, reward, done, False, info

    """"
    def reset(self):

        self.agent_row = self.pregrid_agent_row
        self.agent_col = self.pregrid_agent_col
        self.grid_size = (self.gridsz_num_rows, self.gridsz_num_cols)

        self.agent_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        self.agent_grid[self.pregrid_agent_row][self.pregrid_agent_col] = 1

        self.agent_dir = -1
        self.agent_dir_arr = np.zeros(4)

        self.postgrid_dir = -1
        self.postgrid_dir_arr = np.zeros(4)

        if(self.pregrid_agent_dir == "east"):
            self.agent_dir = 0
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "south"):
            self.agent_dir = 1
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "west"):
            self.agent_dir = 2
            self.agent_dir_arr[self.agent_dir] = 1
        elif(self.pregrid_agent_dir == "north"):
            self.agent_dir = 3
            self.agent_dir_arr[self.agent_dir] = 1

        if(self.postgrid_agent_dir == "east"):
            self.postgrid_dir = 0
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "south"):
            self.postgrid_dir = 1
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "west"):
            self.postgrid_dir = 2
            self.postgrid_dir_arr[self.postgrid_dir] = 1
        elif(self.postgrid_agent_dir == "north"):
            self.postgrid_dir = 3
            self.postgrid_dir_arr[self.postgrid_dir] = 1


        self.walls_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for wall in self.walls:
            self.walls_grid[wall[0]][wall[1]] = 1

        self.markers_grid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for marker in self.pregrid_markers:
            self.markers_grid[marker[0]][marker[1]] = 1

        self.agent_postgrid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        self.agent_postgrid[self.postgrid_agent_row][self.postgrid_agent_col] = 1

        self.markers_postgrid = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols))
        for marker in self.postgrid_markers:
            self.markers_postgrid[marker[0]][marker[1]] = 1

        observation = self._get_obs()
        observation = self.flatten_obs(observation)
        info = self._get_info()

        return observation, info
    """

    def reset(self):
        
        observation = self._get_obs()
        observation = self.flatten_obs(observation)
        info = self._get_info()

        return observation, info



    def render(self, mode='human'):
        dtype = '<U2'
        board = np.zeros((self.gridsz_num_rows, self.gridsz_num_cols), dtype = dtype)
        board[:] = ' '

        board[np.where(self.walls_grid == 1)] = '#'
        board[np.where(self.markers_grid == 1)] = 'M'

        if (self.markers_grid[self.agent_row][self.agent_col] == 1):

            if (self.agent_dir == 0):
                board[self.agent_row][self.agent_col] = 'E'
            elif (self.agent_dir == 1):
                board[self.agent_row][self.agent_col] = 'S'
            elif (self.agent_dir == 2):
                board[self.agent_row][self.agent_col] = 'W'
            elif (self.agent_dir == 3):
                board[self.agent_row][self.agent_col] = 'N'

        else:
            if (self.agent_dir == 0):
                board[self.agent_row][self.agent_col] = 'e'
            elif (self.agent_dir == 1):
                board[self.agent_row][self.agent_col] = 's'
            elif (self.agent_dir == 2):
                board[self.agent_row][self.agent_col] = 'w'
            elif (self.agent_dir == 3):
                board[self.agent_row][self.agent_col] = 'n'

        print(board)



    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()






                

    

    




        


        






