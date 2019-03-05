from opt import opt
import argparse


parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

#%%

img, orig_img, im_name, im_dim_list = data_loader.getitem()