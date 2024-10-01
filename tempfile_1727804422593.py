import ctypes

# 获取大写锁定状态
def get_key_lock_state(key: int):
    r"""获取指定按键锁定状态

    :param key: 按键码
    :return:
    """

    key_lock_state = ctypes.windll.user32.GetKeyState(key)
    return key_lock_state & 0x0001 != 0

# 测试
if get_key_lock_state(20):
    print("大写锁定键已开启")
else:
    print("大写锁定键已关闭")
