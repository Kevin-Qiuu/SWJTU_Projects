import pyrealsense2 as rs

class CameraRewrite(rs.context):
    def __setattr__(self, key, value):
        print('父的属性发生了变化')
        self.__dict__[key] = value