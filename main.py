#import all client and server scripts
import Hw5.module
import Hw5.client
import Hw5.agent
import common
import common.menu_utils

def ModMain():
    Hw5.client.ClientMain()

script_server = common.menu_utils.GetScriptServer()

def ModTick(dt):
    common.startScript("Hw5/menu.py")
    data = script_server.read_data()
    while data:
        Hw5.module.parseInput(data.strip())
        data = script_server.read_data()
