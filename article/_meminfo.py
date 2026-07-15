# -*- coding: utf-8 -*-
import ctypes


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


m = MEMORYSTATUSEX()
m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))

GB = 1024 ** 3
print(f"物理内存总量 : {m.ullTotalPhys / GB:.2f} GB")
print(f"物理内存可用 : {m.ullAvailPhys / GB:.2f} GB")
print(f"内存占用率   : {m.dwMemoryLoad}%")
print(f"页面文件总量 : {m.ullTotalPageFile / GB:.2f} GB")
print(f"页面文件可用 : {m.ullAvailPageFile / GB:.2f} GB")
