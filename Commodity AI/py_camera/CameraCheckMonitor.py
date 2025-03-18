class MonitorVariable:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value,):
        if new_value != self._value:
            print(f"Value changed from {self._value} to {new_value}")
            # 你可以在这里加入文件操作代码
            self._value = new_value

# 使用方式
monitor = MonitorVariable(5)
monitor.value = 10  # 输出：Value changed from 5 to 10
monitor.value = 10  # 没有输出，因为值没有变
