import AGT
import ENV
import os

class trainer():
    def __init__(self):
        pass
    
    def run_train(self, env, agent, opt = 0, text_mode = 0):
        env.update_state()
        state = env.state_list
        # Get action from agent
        if opt == 0:
            action = agent.trainer.get_action(state)
        if opt == 1:
            action = agent.trainer.get_action_test(state)
        # 1 Step of ENV
        info_list = env.step(action, agent.episode)
        
        # Update Reward
        reward = env.get_reward()
        
        # Update Next State
        env.update_state()
        next_state = env.state_list
        
        # memorize
        agent.trainer.remember(state, action, reward, next_state, False)
        
        if env.time % agent.training_interval == 0:
            agent.trainer.train_long_memory()
        
        # Reset environment
        if env.time > agent.episode:
            agent.n_game += 1
            agent.trainer.n_game += 1
            # new High score 
            if(env.whole_reward > agent.high_reward):
                agent.high_reward = env.whole_reward
                if os.path.isfile(('DQN_save/' + agent.model_name)):
                    os.remove(('DQN_save/' + agent.model_name))
                agent.model_name = 'model_' + str(agent.n_game) + '.pth'
                agent.trainer.model.save(agent.model_name)
            train_result = agent.Get_training_record(env.whole_reward)
            env.Reset()
            env.update_state()     
            if text_mode == 1:
                return train_result  
            return [info_list, train_result]

        # Finish!
        if agent.n_game >= agent.trainer.epoch:
            env.Reset()
            return False  

        return info_list