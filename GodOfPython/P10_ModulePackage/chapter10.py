from time import * #time 모듈안에 모든것
import sys, os, math #여러 모듈 기술
import GodOfPython.P10_ModulePackage.sample as sp #별명 사용

#sleep(3)
print(localtime()[0], localtime()[1], localtime()[2])
print(sys.path)
print(os.getcwd())
sp.sample_func()

