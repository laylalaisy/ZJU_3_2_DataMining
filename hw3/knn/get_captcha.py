import urllib

for i in range(1,101):
    file_name = str(i)+".jpg"
    urllib.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", file_name)
