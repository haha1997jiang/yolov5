import os
import ctypes
import pefile

def check_dll_arch(dll_path):
    try:
        # 解析 DLL 文件
        pe = pefile.PE(dll_path)

        # 获取机器类型
        machine_type = pe.FILE_HEADER.Machine


        # 根据机器类型打印架构
        if machine_type == 0x14c:  # IMAGE_FILE_MACHINE_I386
            print("DLL 是 32 位的")
            return False

        elif machine_type == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
            print("DLL 是 64 位的")
            return True
        else:
            print("DLL 的架构未知")
            return False

    except Exception as e:
        print("解析 DLL 失败:", e)
        return False

if __name__ == '__main__':
    # 替换路径为你的实际DLL文件路径
    # dll_path = os.path.join(os.getcwd(), 'msdk32.dll')
    dll_path = os.path.join(os.getcwd(), 'msdk_x64.dll')
    if check_dll_arch(dll_path):
        dll = ctypes.WinDLL(dll_path)
        handle = dll.M_Open(1)
        c = dll.M_ResolutionUsed(handle, 2560, 1440)
        dll.M_MoveTo3(handle, 1280, 720)
        dll.M_Close(handle)

    print("1")
    pass