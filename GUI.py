from select import select
import AGV

import pygame
import tkinter as tk
from tkinter import ttk
import os
import platform
import time

class GUI():
    def __init__(self, env):
        self.rows = 20
        self.width = 500
        self.height = 500
        
        # Main window
        self.root = tk.Tk()  
        self.root.title("AGV Reinforeced Learning Simulator")
        self.root.resizable(False, False)
        self.root.configure(background='#000000')
        self.running = False

        # Large Frame
        self.win_frame = tk.Frame(self.root, width = self.width + 300, height = self.height, 
                                highlightbackground = '#595959', highlightthickness = 2)

        # menu (left side)
        self.menu = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)
        self.menu_label = tk.Label(self.menu, text = 'Control Pannel', font = ("Console", "12"))
        self.Start_button = tk.Button(self.menu, text= "Start", bg = '#728f96', 
                                    font = ("Console", "12"), activebackground='#d45f5f')
        self.Start_button.bind("<Button-1>", self.start_env)
        
        self.Stop_button = tk.Button(self.menu, text= "Stop", bg = '#728f96', 
                                    font = ("Console", "12"), activebackground='#d45f5f')
        self.Stop_button.bind("<Button-1>", self.stop_env)
        
        self.Reset_button = tk.Button(self.menu, text = "Reset", font = ("Console", "12"), 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Reset_button.bind("<Button-1>", self.reset_env)
        
        self.Clear_button = tk.Button(self.menu, text = "Clear Log", font = ("Console", "12"), 
                                    bg = '#728f96', activebackground='#d45f5f')
        self.Clear_button.bind("<Button-1>", self.clear_log)
        
        # Setting(Middle side)
        self.setting = tk.Frame(self.win_frame, width = 200, height = 516, highlightbackground = '#595959', highlightthickness=2)   
        self.setting_label = tk.Label(self.setting, text = 'Setting Pannel', font = ("Console", "12"))   
        
        # Speed setting
        self.speed_var = tk.IntVar()
        self.speed_label = tk.Label(self.setting, text = 'Simulation Speed', font = ("Console", "10"))
        self.speed_scale = tk.Scale(self.setting, variable = self.speed_var, orient="horizontal", state = 'active',
                                    showvalue = True, from_ = 1000, to = 10, length = 200,
                                    highlightbackground = '#728f96', activebackground = '#728f96')
        
        # AGV Algorithm Setting
        self.algorithm_label = tk.Label(self.setting, text = 'Path Finding Algorithm', font = ("Console", "10"))
        self.algorithm_box = ttk.Combobox(self.setting, 
                                    values=["Radom Move", "Deterministic", "Deep Q Network", "DQN Learned model"], state = 'readonly')
        self.algorithm_box.current(0)
        self.algorithm_box.bind("<<ComboboxSelected>>", self.algorithm_changed)
        
        # State (Right side)
        self.state = tk.Frame(self.win_frame, width = 400, height = 516 / 2, highlightbackground = '#595959', highlightthickness=2)   
        self.state_label = tk.Label(self.state, text = 'State Pannel', font = ("Console", "12"))  
        
        self.state_scroll = tk.Scrollbar(self.state, orient='vertical')
        self.state_box = tk.Listbox(self.state, yscrollcommand = self.state_scroll.set, width = 400, height = 400)
        self.state_scroll.config(command=self.state_box.yview)
        
        # Log (Right side)
        self.log = tk.Frame(self.win_frame, width = 400, height = 516 / 2, highlightbackground = '#595959', highlightthickness=2)   
        self.log_label = tk.Label(self.log, text = 'Log Pannel', font = ("Console", "12")) 
        
        self.log_scroll = tk.Scrollbar(self.log, orient='vertical')
        self.log_box = tk.Listbox(self.log, yscrollcommand = self.log_scroll.set, width = 400, height = 400)
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
        
        self.setting.pack(side = "left", anchor = 'n')
        self.setting_label.pack()
        self.speed_label.pack()
        self.speed_scale.pack()
        self.algorithm_label.pack()
        self.algorithm_box.pack()
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
        
        # Load simulation environment
        self.env = env
        
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
    
    def run_env(self, event = None):
        if self.running:
            self.make_state_info(self.env.Run())
            self.redrawWindow(self.env.Get_Obj())
        # After 1 second, call run_env again (create a recursive loop)
        self.root.after(self.speed_var.get(), self.run_env)
    
    # If start button is clicked
    def start_env(self, event = None):
        self.running = True
        self.append_log('Start Simulation')
    
    # If stop button is clicked
    def stop_env(self, event = None):
        self.running = False
        self.append_log('Stop Simulation')

    # If reset button is clicked
    def reset_env(self, event = None):
        self.env.Reset()
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
            self.env.running_opt = 0
        if event.widget.get() == "Deterministic":
            self.env.running_opt = 1
        if event.widget.get() == "Deep Q Network":
            self.env.running_opt = 2
        if event.widget.get() == "DQN Learned model":
            self.env.running_opt = 3
            
    def make_state_info(self, state_list):
        self.state_box.delete(0, self.state_box.size())
        state_text = ""
        state_text += "                    AGV1    AGV2    AGV3"
        self.update_state(state_text)
        state_text = ""
        state_text += "Position: "
        state_text += str(state_list[0][0]) + " " + str(state_list[0][1]) + " " + str(state_list[0][2])
        self.update_state(state_text)
        state_text = ""
        state_text += "Load:          "
        state_text += str(state_list[2][0]) + "  " + str(state_list[2][1]) + "  " + str(state_list[2][1])
        self.update_state(state_text)
        state_text = ""
        state_text += "Machines: "
        state_text += str(state_list[3][0]) + "  " + str(state_list[3][1]) + "  " + str(state_list[3][1])
        self.update_state(state_text)
        state_text = ""
        state_text += "Products:        "
        state_text += str(self.env.Get_product()[0]) + "        " + str(self.env.Get_product()[1]) + "        " + str(self.env.Get_product()[2]) 
        self.update_state(state_text)
        state_text = ""
        self.update_state(state_text)
        state_text = ""
        state_text += "Throuput(products/time): "
        state_text += str(self.env.Get_throuput())
        self.update_state(state_text)
        return 