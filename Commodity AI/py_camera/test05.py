import keyboard
import time

def myfun():
    print("hello")

keyboard.on_press_key("enter",myfun)

keyboard.wait("esc")