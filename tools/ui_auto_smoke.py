import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import ImageGrab
from stratamatch.gui import Application,SectionEditor,configure_dpi
from stratamatch.io import local,read

configure_dpi();root=tk.Tk();app=Application(root)
section=read('outputs/layout-validation.json')['proposal'];app.sections=[section];app.refresh()
editor=SectionEditor(root,section,lambda _:None);editor.geometry('1100x850+30+30')
def capture():
    editor.update_idletasks();x,y=editor.winfo_rootx(),editor.winfo_rooty()
    ImageGrab.grab(bbox=(x,y,x+editor.winfo_width(),y+editor.winfo_height())).save(local('outputs/gui_smoke/automatic_layout.png'))
    root.destroy()
root.after(900,capture);root.mainloop()
