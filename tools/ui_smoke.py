"""Capture only this application's windows and exercise its real run button."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import ImageGrab
from stratamatch.gui import Application,SectionEditor,configure_dpi
from stratamatch.io import local,write

configure_dpi()
root=tk.Tk();app=Application(root);app.load_demo();root.geometry('1200x850+20+20')
out=local('outputs/gui_smoke');out.mkdir(parents=True,exist_ok=True)
def capture(window,name):
    window.update_idletasks()
    x,y=window.winfo_rootx(),window.winfo_rooty()
    ImageGrab.grab(bbox=(x,y,x+window.winfo_width(),y+window.winfo_height())).save(out/name)
def settings():
    capture(root,'main.png');app.notebook.select(1);root.after(700,editor)
def editor():
    capture(root,'settings.png')
    modal=SectionEditor(root,app.sections[0],lambda _:None)
    def finish():capture(modal,'editor.png');modal.destroy();start()
    root.after(700,finish)
def start():
    app.output.set(str(out));app.start();root.after(300,done)
def done():
    if app.worker.is_alive() or not app.last_result:root.after(300,done);return
    capture(root,'results.png');write(out/'smoke.json',app.last_result);root.destroy()
root.after(700,settings);root.after(30000,root.destroy);root.mainloop()
