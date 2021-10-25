import shutil
from datetime import datetime
import os

def set_up_run_folder():
    # Removes the running folder and moves it to the folder "old\_runs". Sets up all folders.

    try:
        os.system("mkdir " + "old_outputs -p")
    except:
        pass
    try:
        shutil.move("log", "old_outputs/" + datetime.now().strftime("%d%m%Y%H%M%S"))  
        shutil.move("plots", "old_outputs/" + datetime.now().strftime("%d%m%Y%H%M%S"))  

    except:
        pass
    try:
        os.system("mkdir " + "log -p")
        os.system("mkdir " + "plots -p")

    except:
        pass


