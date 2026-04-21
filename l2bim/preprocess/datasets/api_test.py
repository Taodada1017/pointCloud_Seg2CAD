import requests

# 选择要调用的接口地址（三选一）
#url = "http://127.0.0.1:8000/preview"  # 预览,不能改变参数 暂时弃用
url1 = "http://127.0.0.1:8000/preview/v2"  # 预览_v2,可以改变参数
url2 = "http://127.0.0.1:8000/detect/once"  # 一次检测
url3 = "http://127.0.0.1:8000/detect/multi"  # 分步检测
url4 = "http://127.0.0.1:8000/projection"  # 投影
url=[url4]

# 替换为你的.pcd文件路径
pcd_path=r"C:\Users\Server\xwechat_files\wxid_649z3170r5rz22_2353\msg\file\2025-11\Preview10.27_5_noreflect.pcd"
cfg_path = r"D:\work\l2bim\configs\interval\15m\1F\1f_office_03.yaml"
files = {"file": open(pcd_path, "rb")}
data = {"cfg": cfg_path, "save_png": "true"}
# 发送请求并打印结果
#res = requests.post(url, files=files,data=data)  #两个参数，pcd和yaml
print(f"1.{url1}点云预览API")
res = requests.post(url1, files=files)
# 关闭文件
files["file"].close()

# 先打印原始返回内容，确认格式
print("接口原始返回：", res.text)

try:
    # 尝试解析JSON
    data = res.json()
    print("JSON解析结果：", data)
except requests.exceptions.JSONDecodeError:
    print("接口返回不是JSON格式，无需解析")

# files = {"file": open(pcd_path, "rb")}
# data = {'cfg': cfg_path, 'return_numpy': True, 'return_linesegs': True}
# print(f"2.{url2}检测线段API")
# res = requests.post(url2, files=files,data=data)

# # 先打印原始返回内容，确认格式
# print("接口原始返回：", res.text)

# try:
#     # 尝试解析JSON
#     data = res.json()
#     print("JSON解析结果：", data)
# except requests.exceptions.JSONDecodeError:
#     print("接口返回不是JSON格式，无需解析")

for i in range(len(url)):
    print(f"{i+3}.{url[i]}")
    files = {"file": open(pcd_path, "rb")}
    data = {"cfg": cfg_path, "save_png": "true", "projection_scale": 200.0}
    res = requests.post(url[i], files=files,data=data)
    files["file"].close()
    print("接口原始返回：", res.text)
    try:
        # 尝试解析JSON
        data = res.json()
        print("JSON解析结果：", data)
    except requests.exceptions.JSONDecodeError:
        print("接口返回不是JSON格式，无需解析")