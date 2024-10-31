# - 定义Log类并使用单例模式
class Log:
    # ANSI 转义序列，用于蓝色文本输出
    INDENT = 0
    BLUE = '\033[94m'
    RESET = '\033[0m'  # 用于重置颜色


    def __init__(self,env = None):
        self.env = env

    def set_env(self,env):
        self.env = env
    def increase_indent(self):
        Log.INDENT += 1

    def decrease_indent(self):
        if Log.INDENT > 0:
            Log.INDENT -= 1

    def log(self, message):
        if self.env is not None:
            print(Log.BLUE + "  " * Log.INDENT +  self.env + ": " + message + Log.RESET)
        else:
            print(" " * Log.INDENT + Log.BLUE + message + Log.RESET)