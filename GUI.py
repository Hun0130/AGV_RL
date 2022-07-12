from distutils.log import fatal
from select import select

import pygame
import pyglet
import tkinter as tk
from tkinter import ttk
import os
import platform
from tkinter import font
import threading
from tkinter import filedialog
import torch

class GUI():
    def __init__(self, env, agent, trainer):
        self.rows = 20
        self.width = 500
        self.height = 500
        
        # Load simulation environment
        self.env = env
        self.agent = agent
        self.trainer = trainer
        
        # Main window
        self.root = tk.Tk()  
        pyglet.font.add_file('D2Coding.ttf')
        self.root.title("AGV Reinforeced Learning Simulator")
        self.root.resizable(False, False)
        self.root.configure(background='#000000')
        
        # IF GUI mode is running
        self.running_check = False
        # IF text training model is running
        self.training_check = False
        # IF trained model is saved
        self.saving_check = False
        
        # font option
        self.root.option_add('*Dialog.msg.font', 'D2Coding Nerd Font 12')
        self.font_style1 = ('D2Coding Nerd Font', 14)
        self.font_style2 = ('D2Coding Nerd Font', 10)
        
        # Large Frame
        self.win_frame = tk.Frame(self.root, width = self.width + 300, height = self.height, 
                                highlightbackground = '#595959', highlightthickness = 2)

        # menu (left side)
        self.menu = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)
        self.menu_label = tk.Label(self.menu, text = 'Control Pannel', font = self.font_style1)
        self.Start_button = tk.Button(self.menu, text= "Start", bg = '#728f96', 
                                    font = self.font_style1, activebackground='#d45f5f')
        self.Start_button.bind("<Button-1>", self.start_env)
        
        self.Stop_button = tk.Button(self.menu, text= "Stop", bg = '#728f96', 
                                    font = self.font_style1, activebackground='#d45f5f')
        self.Stop_button.bind("<Button-1>", self.stop_env)
        
        self.Reset_button = tk.Button(self.menu, text = "Reset", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Reset_button.bind("<Button-1>", self.reset_env)
        
        self.Clear_button = tk.Button(self.menu, text = "Clear Log", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Clear_button.bind("<Button-1>", self.clear_log)
        
        self.Learn_button = tk.Button(self.menu, text = "Text Mode", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Learn_button.bind("<Button-1>", self.training)
        
        self.Load_Model_button = tk.Button(self.menu, text = "Load Model", font = self.font_style1, 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Load_Model_button.bind("<Button-1>", self.load)
        
        # Setting(Middle side)
        self.setting = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)   
        self.setting_label = tk.Label(self.setting, text = 'Setting Pannel', font = self.font_style1)   
        
        # Speed setting
        self.speed_var = tk.IntVar()
        self.speed_label = tk.Label(self.setting, text = 'Simulation Speed', font = self.font_style2)
        self.speed_scale = tk.Scale(self.setting, variable = self.speed_var, orient="horizontal", state = 'active',
                                    showvalue = True, from_ = 1000, to = 1, length = 200,
                                    highlightbackground = '#728f96', activebackground = '#728f96', font=self.font_style2)
        self.speed_scale.set(1000)
        
        # AGV Algorithm Setting
        self.algorithm_label = tk.Label(self.setting, text = 'Path Finding Algorithm', font = self.font_style2)
        self.algorithm_box = ttk.Combobox(self.setting, 
                                    values=["Radom Move", "Deterministic", "Deep Q Network", "DQN Learned model"], state = 'readonly',
                                    font=self.font_style2)
        self.algorithm_box.current(0)
        self.algorithm_box.bind("<<ComboboxSelected>>", self.algorithm_changed)
        
        # Training setting
        self.parameter_label = tk.Label(self.setting, text='Hyper Parameters', font = self.font_style2)
        
        self.lr_var = tk.StringVar()
        self.lr_label = tk.Label(self.setting, text='Learning Rate [0 1)', font = self.font_style2)
        self.lr_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.lr_var)
        self.lr_entry.insert(0, str(self.agent.trainer.learning_rate))
        self.lr_var.trace('w', self.change_lr)
        
        self.gamma_var = tk.StringVar()
        self.gamma_label = tk.Label(self.setting, text='Gamma Rate (0 1)', font = self.font_style2)
        self.gamma_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.gamma_var)
        self.gamma_entry.insert(0, str(self.agent.trainer.gamma))
        self.gamma_var.trace('w', self.change_gamma)
        
        self.episode_var = tk.StringVar()
        self.episode_label = tk.Label(self.setting, text='Episode', font = self.font_style2)
        self.episode_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.episode_var)
        self.episode_entry.insert(0, str(self.agent.trainer.MAX_MEMORY))
        self.episode_var.trace('w', self.change_episode)
        
        self.epoch_var = tk.StringVar()
        self.epoch_label = tk.Label(self.setting, text='Epoch', font = self.font_style2)
        self.epoch_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.epoch_var)
        self.epoch_entry.insert(0, str(self.agent.trainer.epoch))
        self.epoch_var.trace('w', self.change_epoch)
        
        self.batch_size_var = tk.StringVar()
        self.batch_size_label = tk.Label(self.setting, text='Batch Size', font = self.font_style2)
        self.batch_size_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.batch_size_var)
        self.batch_size_entry.insert(0, str(self.agent.trainer.BATCH_SIZE))
        self.batch_size_var.trace('w', self.change_batch_size)
        
        self.step_interval_var = tk.StringVar()
        self.step_interval_label = tk.Label(self.setting, text='Step Interval', font = self.font_style2)
        self.step_interval_entry = tk.Entry(self.setting, width = 30, bg = '#728f96', font=self.font_style2, textvariable=self.step_interval_var)
        self.step_interval_entry.insert(0, str(self.agent.training_interval))
        self.step_interval_var.trace('w', self.change_step_interval)
        
        
        # State (Right side)
        self.state = tk.Frame(self.win_frame, width = 400, height = 516 / 2, highlightbackground = '#595959', highlightthickness=2)   
        self.state_label = tk.Label(self.state, text = 'State Pannel', font = self.font_style1)  
        
        self.state_scroll = tk.Scrollbar(self.state, orient='vertical')
        self.state_box = tk.Listbox(self.state, yscrollcommand = self.state_scroll.set, width = 400, height = 400)
        self.state_scroll.config(command=self.state_box.yview)
        
        # Log (Right side)
        self.log = tk.Frame(self.win_frame, width = 400, height = 516 / 2, highlightbackground = '#595959', highlightthickness=2)   
        self.log_label = tk.Label(self.log, text = 'Log Pannel', font = self.font_style1) 
        self.log_scroll = tk.Scrollbar(self.log, orient='vertical')
        self.log_box = tk.Listbox(self.log, yscrollcommand = self.log_scroll.set, width = 400, height = 400, font=self.font_style2)
        self.log_scroll.config(command=self.log_box.yview)
        
        # Start log
        self.append_log('AGV Reinforeced Learning Simulator - CSI Lab')
        
        # pygame
        self.pygame_frame = tk.Frame(self.win_frame, width = self.width, height = self.height, 
                                    highlightbackground='#595959', highlightthickness=2)
        self.embed = tk.Frame(self.pygame_frame, width = self.width, height = self.height)

        # Packing
        self.win_frame.pack(expand = True)
        self.win_frame.pack_propagate(0)

        self.menu.pack(side="left")
        self.menu.pack_propagate(0)
        self.menu_label.pack()
        
        self.Start_button.pack(ipadx = 60)
        self.Stop_button.pack(ipadx = 60)
        self.Reset_button.pack(ipadx = 60)
        self.Clear_button.pack(ipadx= 60)
        self.Learn_button.pack(ipadx=60)
        self.Load_Model_button.pack(ipadx=60)
        
        self.setting.pack(side = "left", anchor = 'n')
        self.setting_label.pack()
        self.speed_label.pack()
        self.speed_scale.pack()
        self.algorithm_label.pack()
        self.algorithm_box.pack()
        self.parameter_label.pack(pady= 3)
        self.lr_label.pack()
        self.lr_entry.pack()
        self.gamma_label.pack()
        self.gamma_entry.pack()
        self.episode_label.pack()
        self.episode_entry.pack()
        self.epoch_label.pack()
        self.epoch_entry.pack()
        self.batch_size_label.pack()
        self.batch_size_entry.pack()
        self.step_interval_label.pack()
        self.step_interval_entry.pack()
        self.setting.pack_propagate(0)
        
        self.state.pack()
        self.state_label.pack()
        self.state_box.pack()
        self.state.pack_propagate(0)
        
        self.log.pack(side = "left")
        self.log_label.pack()
        self.log_box.pack()
        self.log.pack_propagate(0)
        
        # self.pygame_frame.pack(side="left")
        # self.embed.pack()
        
        # This embeds the pygame window
        os.environ['SDL_WINDOWID'] = str(self.embed.winfo_id())
        system = platform.system()
        if system == "Windows":
            os.environ['SDL_VIDEODRIVER'] = 'windib'
        elif system == "Linux":
            os.environ['SDL_VIDEODRIVER'] = 'x11'

        self.root.update_idletasks()
        
        # Start pygame
        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        self.redrawWindow(self.env.Get_Obj())
        self.root.after(1000, self.run_env())
        self.root.mainloop()
    
    # Draw Grid with white line and black backgrounds
    def drawGrid(self, w, surface):
        sizeBtwn = w // self.rows
        x = 0
        y = 0
        for l in range(self.rows):
            x = x + sizeBtwn
            y = y + sizeBtwn
            pygame.draw.line(surface, (255, 255, 255), (x, 0),(x, w))
            pygame.draw.line(surface, (255, 255, 255), (0, y),(w, y))

    # Update windows
    def redrawWindow(self, obj_list):
        # self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Simulation State')
        self.win.fill((0,0,0))
        
        self.drawGrid(self.width, self.win)
        
        for obj in obj_list:
            obj.draw(self.win)
            
        # pygame.display.update()
        pygame.display.flip()
        return

    # Run environment
    def run_env(self, event = None):
        if self.running_check:
            self.env.update_state()
            run = self.env.step(self.agent.action(self.env.Get_AGV(), self.env.Get_Buffer(), 
                                                self.env.Get_Machine(), self.env.state_list), self.agent.episode)
            if self.env.time > self.agent.episode:
                self.running_check = False
            self.make_state_info(run)
            self.redrawWindow(self.env.Get_Obj())
        # After <speed_var> second, call run_env again (create a recursive loop)
        self.root.after(self.speed_var.get(), self.run_env)
    
    # Training without GUI update
    def training(self, event = None):
        self.running_check = False
        self.training_check = True
        self.agent.running_opt = 2
        self.thread = threading.Thread(target = self.training_loop)
        self.thread_daemon = True
        self.thread.start()
    
    # Training with no GUI
    def training_loop(self, event = None):
        self.append_log(('learning_rate: ' + str(self.agent.trainer.learning_rate) + ' gamma: ' + str(self.agent.trainer.gamma)))
        self.append_log('episode: ' + str(self.agent.trainer.MAX_MEMORY) + ' epoch: ' + str(self.agent.trainer.epoch))
        self.append_log('batch size: ' + str(self.agent.trainer.BATCH_SIZE) + ' step interval : ' + str(self.agent.training_interval))
        while True:
            result = self.trainer.run_train(self.env, self.agent, opt = 0, text_mode = 1)
            if result == False:
                break
            elif type(result) == list:
                continue
            else:
                self.append_log(result)
        self.training_check = False
        self.agent.running_opt = 0
        # self.reset_env()
        
    # change learning rate value
    def change_lr(self, *args):
        if self.training_check:
            return
        if self.running_check:
            return
        try:
            self.agent.trainer.learning_rate = float(self.lr_var.get())
        except:
            return
        
    # change gamma value
    def change_gamma(self, *args):
        if self.training_check:
            return
        if self.running_check:
            
            return
        try:
            self.agent.trainer.gamma = float(self.gamma_var.get())
        except:
            return
        
    # change episode value
    def change_episode(self, *args):
        if self.training_check:
            return
        if self.running_check:
            return
        try:
            self.agent.trainer.MAX_MEMORY = int(self.episode_var.get())
        except:
            return
        
    # change epoch value
    def change_epoch(self, *args):
        if self.training_check:
            return
        if self.running_check:
            return
        try:
            self.agent.trainer.epoch = int(self.epoch_var.get())
        except:
            return
        
    # change batch_size value
    def change_batch_size(self, *args):
        if self.training_check:
            return
        if self.running_check:
            return
        try:
            self.agent.trainer.BATCH_SIZE = int(self.batch_size_var.get())
        except:
            return
        
    # change step interval value
    def change_step_interval(self, *args):
        if self.training_check:
            return
        if self.running_check:
            return
        try:
            self.agent.training_interval = int(self.step_interval_var.get())
        except:
            return
    
    # If start button is clicked
    def start_env(self, event = None):
        if self.training_check:
            return
        self.running_check = True
        self.append_log('Start Simulation')
    
    # If stop button is clicked
    def stop_env(self, event = None):
        if self.training_check:
            return
        self.running_check = False
        self.append_log('Stop Simulation')

    # If reset button is clicked
    def reset_env(self, event = None):
        if self.training_check:
            return
        self.make_state_info(self.env.Reset())  
        self.redrawWindow(self.env.Get_Obj()) 
        self.append_log('Reset Simulation') 
    
    # Append Log
    def append_log(self, msg):
        self.log_box.insert(tk.END, "{}".format(msg))
        self.log_box.update()
        self.log_box.see(tk.END)

        # Append Log
    def update_state(self, msg):
        self.state_box.insert(tk.END, "{}".format(msg))
        self.state_box.update()
        self.state_box.see(tk.END)
    
    # Clear all Log
    def clear_log(self, event = None):
        self.log_box.delete(0, self.log_box.size())
        self.log_box.see(tk.END)

    # When trajectory algorithm is changed
    def algorithm_changed(self, event):
        self.append_log("Changed trajectory algorithm to {}".format(event.widget.get()))
        if event.widget.get() == "Radom Move":
            self.agent.running_opt = 0
        if event.widget.get() == "Deterministic":
            self.agent.running_opt = 1
        if event.widget.get() == "Deep Q Network":
            self.agent.running_opt = 2
            self.append_log(('learning_rate: ' + str(self.agent.trainer.learning_rate) + ' gamma: ' + str(self.agent.trainer.gamma)))
            self.append_log('episode: ' + str(self.agent.trainer.MAX_MEMORY) + ' epoch: ' + str(self.agent.trainer.epoch))
            self.append_log('batch size: ' + str(self.agent.trainer.BATCH_SIZE) + ' step interval : ' + str(self.agent.training_interval))
        if event.widget.get() == "DQN Learned model":
            if self.saving_check == True:
                self.agent.running_opt = 3
            else:
                self.append_log("Please Load Model First!")
                
            
    def make_state_info(self, info_list):
        if info_list == False:
            return
        
        if len(info_list) == 2:
            self.append_log(info_list[1])
            info_list = info_list[0]

        self.state_box.delete(0, self.state_box.size())
        self.update_state('{:<15} {:<10} {:<10} {:<10}'.format('Machine:', 'AGV1', 'AGV2', 'AGV3'))
        self.update_state('{:<15} {:<10} {:<10} {:<10}'.format('Position:', str(info_list[0][0]), str(info_list[0][1]), str(info_list[0][2])))
        self.update_state('{:<15} {:<10} {:<10} {:<10}'.format('Load Up:', str(bool(info_list[2][0])), str(bool(info_list[2][1])), str(bool(info_list[2][2]))))
        self.update_state('{:<15} {:<10} {:<10} {:<10}'.format('Load Down:', str(bool(info_list[3][0])), str(bool(info_list[3][1])), str(bool(info_list[3][1]))))
        self.update_state('{:<15} {:<10} {:<10} {:<10}'.format('Products:', str(self.env.products_num[0]), str(self.env.products_num[1]), str(self.env.products_num[2])))
        self.update_state("")
        self.update_state('{:<20}{:^10}'.format("Throughput(products/time):", str(self.env.Get_throuput())))
        return 

    def load(self, event = None):
        filename = filedialog.askopenfilename(initialdir="DQN_save/", title = "Select saved model",
                                        filetypes=(("pth files", "*.pth"),
                                        ("all files", "*.*")))
        try: 
            self.agent.trainer.model = torch.load(filename)
            self.saving_check = True
        except:
            self.saving_check = False