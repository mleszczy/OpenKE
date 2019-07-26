import config
from models import *
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_use_gpu(True)
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.
con.set_result_dir("./result")
con.set_val_link(False)
con.set_test_link(True)
# con.set_test_triple(True)
con.init()
con.set_test_model(TransE)
con.test()
