# keyboard_handler.py
from pynput.keyboard import Key, Listener

def on_press(key):
    try:
        if key == Key.up:
            print('Up Arrow')
        elif key == Key.down:
            print('Down Arrow')
        elif key == Key.left:
            print('Left Arrow')
        elif key == Key.right:
            print('Right Arrow')
    except AttributeError:
        print(f"Key {key} pressed")

def start_listener():
    with Listener(on_press=on_press) as listener:
        listener.join()
