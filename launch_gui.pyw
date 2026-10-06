"""Double-click entry; keep diagnostics and temporary files inside this project."""
import os
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
project=root.parent
os.chdir(root)
temp=project/'输出数据'/'临时';temp.mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(temp)
sys.dont_write_bytecode=True
try:
    from stratamatch.gui import main
    main()
except Exception:
    import traceback
    logs=project/'输出数据'/'日志';logs.mkdir(parents=True,exist_ok=True)
    (logs/'gui_error.log').write_text(traceback.format_exc(),encoding='utf-8')
    import tkinter.messagebox
    tkinter.messagebox.showerror('界面启动失败','请查看 输出数据/日志/gui_error.log')
