import time
import inspect


def timer():
    start = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。

    def end(method_name="Unnamed function"):
        print(method_name + " took : " + str(time.time() - start) + " seconds.")
        return  # return none

    return end
