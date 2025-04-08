import numpy as np


class RewardGenerator:
    def __init__(self, capacities, m):
        self.capacities = capacities
        self.m = m

        self.D_t_prev = None # this stores the previous iteration's value for D_t, which is used in the reward calculation. 
    
    def reset(self):
        self.D_t_prev = None

    def paper_reward(self,observation, migrate):
            L = np.sum(observation, axis=1)
            B = L / self.capacities
            B_bar = np.sum(B) / self.m
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1
            
            # compute the improvement in controller load after migration.
            D_t_diff = 0
            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t

            # get the migration cost
            # F = self.migration_cost

            reward = 0
            if migrate:
                # reward = D_t_diff / F
                reward = D_t_diff

            # print(observation)
            # print("load_ratios: ", B)
            # print("D_t: ", D_t)
            # print("D_t_diff: ", D_t_diff)
            # print(reward)
            
            return reward

    def penalize_poor_inaction_reward(self, observation, migrate):
            '''
            In this reward, we want to 
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            D_t_diff = 0
            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t

            reward = 0
            # in this example we want to punish not taking an action that would have improved things. 
            # that is to say, if D_t_diff is negative, and you chose not to migrate, then you lose points
            if migrate == False and D_t_diff < 0:
                reward = D_t_diff
                reward = reward ** 2 * -1
            else:
                # otherwise, compute reward normally
                reward = D_t_diff

            return reward

    def penalize_and_encourage_reward(self, observation, migrate):
            '''
            In this reward, we want to both penalize not taking an action and getting a bad score
            and we want to encourage taking an action and getting a good score. 
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            D_t_diff = 0
            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t

            reward = 0
            # in this example we want to punish not taking an action that would have improved things. 
            # that is to say, if D_t_diff is negative, and you chose not to migrate, then you lose points
            if migrate == False and D_t_diff < 0:
                reward = D_t_diff
                reward = reward ** 2 * -1
            elif migrate == True and D_t_diff > 0:
                reward = D_t_diff
                reward = reward ** 2
            elif migrate == False:
                reward = 0
            else:
                reward = D_t_diff

            if migrate: 
            # print out info
                print(observation)
                print("load_ratios: ", B)
                print("average load: ", B_bar)
                print("D_t (degree of balancing): ", D_t)
                print("D_t diff", D_t_diff)
                print("reward: ", reward)

            return reward

    def binary_reward(self, observation, migrate):
            '''
            In this reward, we want to give points only for good decisions.
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            D_t_diff = 0
            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t

            reward = 0
            if migrate == True and D_t_diff > 0:
                reward = D_t_diff * 100

            return reward

    def encourage_explore(self, observation, migrate):
            '''
            In this reward, we want to both penalize not taking an action and getting a bad score
            and we want to encourage taking an action and getting a good score. 
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            D_t_diff = 0

            D_t_prev_temp = self.D_t_prev

            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t

            reward = 0
            # in this example we want to punish not taking an action that would have improved things. 
            # that is to say, if D_t_diff is negative, and you chose not to migrate, then you lose points

            # print("D_t: ", D_t)
            if migrate: 
                # print(observation)
                # print("load_ratios: ", B)
                # print("D_t_prev_temp: ", D_t_prev_temp)
                # print("D_t: ", D_t)
                # print("D_t_diff: ", D_t_diff)
                if D_t_diff > 0:
                    # if good
                    reward = D_t_diff*1000
                else:
                    reward = D_t_diff * 10
                # print("reward: ", reward)
            else:
                # no migration
                reward = -10000

            # if migrate: 
            # # print out info
            #     print(observation)
            #     print("load_ratios: ", B)
            #     print("average load: ", B_bar)
            #     print("D_t (degree of balancing): ", D_t)
            #     print("D_t diff", D_t_diff)
            #     print("reward: ", reward)

            return reward

    def balance(self, observation, migrate):
            '''
            Just promote balance between controllers.
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            # D_t_diff = 0

            # D_t_prev_temp = self.D_t_prev

            # if self.D_t_prev != None:
            #     D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
            #     # print("D_t diff", reward)
            # self.D_t_prev = D_t
            reward = -1 * D_t # the higher the average load, the worse the reward
            # print(reward)
            return reward

    def custom_reward_original(self, observation, migrate):
            '''
            In this reward, we want to reward having a low average load of controllers, and promote good swaps. 
            '''
            # compute Lh(t) by summing each row of the state matrix
            L = np.sum(observation, axis=1)
            # print("L: ", L)

            # compute Bh(t) by dividing each by uh (capacities)
            B = L / self.capacities
            # print("load_ratios: ", B)

            # compute B_bar (average load) by 
            B_bar = np.sum(B) / self.m
            # print("average load: ", B_bar)

            # compute the controller load balancing rate, D(t)
            D_t = 0
            numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
            if B_bar != 0:
                D_t = numerator / B_bar  # Final computation
                # print("D_t (degree of balancing): ", D_t)
            else:
                D_t = 1

            D_t_diff = 0
            
            if self.D_t_prev != None:
                D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
                # print("D_t diff", reward)
            self.D_t_prev = D_t


            # reward calculation. 

            # if they make a good decision give a positive reward
            # if they make a bad decision, give a small negative reward
            # if they do nothing, and things get better, no reward
            # if they do nothing, and things get worse, give a bad reward

            reward = 0
            if migrate:
                if D_t_diff > 0:
                    reward = 100
                elif D_t_diff < 0:
                    reward = -1
                else:
                    reward = 0
            else:
                if D_t_diff > 0:
                    reward = 0
                elif D_t_diff < 0:
                    reward = -100
                else:
                    reward = 0

            # reward = D_t_diff # the higher the average load, the worse the reward
            # print(reward)
            return reward

    def custom_reward(self, observation, migrate):
        '''
        In this reward, we want to reward having a low average load of controllers, and promote good swaps. 
        '''
        # compute Lh(t) by summing each row of the state matrix
        L = np.sum(observation, axis=1)
        # print("L: ", L)

        # compute Bh(t) by dividing each by uh (capacities)
        B = L / self.capacities
        # print("load_ratios: ", B)

        # compute B_bar (average load) by 
        B_bar = np.sum(B) / self.m
        # print("average load: ", B_bar)

        # compute the controller load balancing rate, D(t)
        D_t = 0
        numerator = np.sqrt(np.sum((B - B_bar) ** 2) / self.m)  # Compute standard deviation
        if B_bar != 0:
            D_t = numerator / B_bar  # Final computation
            # print("D_t (degree of balancing): ", D_t)
        else:
            D_t = 1

        D_t_diff = 0
        
        if self.D_t_prev != None:
            D_t_diff = self.D_t_prev - D_t # positive reward for the D_t getting smaller each iteration.
            # print("D_t diff", reward)
        self.D_t_prev = D_t


        # reward calculation. 

        # if they make a good decision give a positive reward
        # if they make a bad decision, give a small negative reward
        # if they do nothing, and things get better, no reward
        # if they do nothing, and things get worse, give a bad reward

        # print("run\n\n")

        reward = 0
        if migrate:
            if D_t_diff > 0:
                reward = 100 * D_t_diff
                # print("D_t_diff: ", D_t_diff)
            elif D_t_diff < 0:
                reward = 100 * D_t_diff # which will be negative
                # print("lost points for poor migration")
            else:
                reward = 0
        else:
            # print("D_t_diff: ", D_t_diff)
            if D_t_diff > 0:
                reward = 0
            elif D_t_diff < 0:
                reward = 200* D_t_diff
            else:
                reward = 0

        # print("D_t_diff: ", D_t_diff)
        # print("reward: ", reward)

        # reward = D_t_diff # the higher the average load, the worse the reward
        # print(reward)
        return reward