import numpy as np

'''
max_win = 10
If time t is predicted as 1, extend the predction 1 to t+1 ~ t+10.
[i.e: NEXT 10 prediction extended as ones (the 10 does not include current t)] 
Same goes for min_win
'''

def quad_win(min_win=5, max_win=30, list_len=8640):
	'''
	min_pt = (m,n), max_pt = (k,q)
	a = (n-q)/(m^2-k*m)
	b = -a*k
	c = q
	'''
	k = list_len-1
	q = max_win
	m = k/2
	n = min_win

	a = (n-q)/(m**2-k*m)
	b = -a*k
	c = q

	t = np.arange(list_len)
	win_lim = np.round(a*(t**2) + b*t + c).astype(int)
	return win_lim

def V_win(min_win=5, max_win=30, list_len=8640):
	'''
	max win - min win find y_intercept
	define negative graph
	absolute it
	shift it back by min win
	'''
	y_in = max_win-min_win
	x_in = list_len/2
	grad = -y_in/x_in
	t = np.arange(list_len)
	win_lim = np.round(abs(grad*t + y_in) + min_win).astype(int)
	return win_lim

def cos_win(min_win=5, max_win=30, list_len=8640):
	'''
	define t len within 2*pi
	cos graph with desired win range
	shift 
	'''
	min_win = min_win * 360
	max_win = max_win * 360

	win_range = max_win - min_win
	t = np.linspace(0, 2*np.pi, list_len)
	win_lim = np.round((win_range/2)*np.cos(t) + (win_range/2) + min_win).astype(int)
	return win_lim


# ==== Usage ====
if __name__ == '__main__':


	import pandas as pd
	import argparse
	import os
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser(description="Description")
	parser.add_argument('-path','--path', default='AA', type=str, help='path of stored data') # Stop at house level, example G:\H6-black\
	parser.add_argument('-save_location', '--save', default='', type=str, help='location to store files (if different from path')
	# parser.add_argument('-hub', '--hub', default='', type=str, help='if only one hub... ')
	parser.add_argument('-hub', '--hub', default="", nargs="+", type=str, help='if only one hub... ') # Example: python time_window.py -hub BS2 BS3
	parser.add_argument('-start_index','--start_date_index', default=0, type=int, help='Processing START Date index')


	args = parser.parse_args()
	path = args.path
	save_path = args.save if len(args.save) > 0 else path
	home_system = os.path.basename(path.strip('/'))
	H = home_system.split('-')
	H_num, color = H[0], H[1][0].upper()
	hubs = args.hub
	# hubs = [args.hub] if len(args.hub) > 0 else sorted(mylistdir(path, bit=f'{color}S', end=False))
	print(f'List of Hubs: {hubs}')

	hub = hubs[0]
	read_root_path = os.path.join(path, "Inference_DB", hub, 'img_inf', 'processed','2019-10-09.csv')

	path = 'C:/Users/Sin Yong Tan/Desktop/to_maggie/H6-black/Inference_DB/BS2/img_inf/processed/2019-10-09.csv'
	
	
	data = pd.read_csv(read_root_path, index_col=0)

	# For Testing the window
	data["occupied"].iloc[:] = 0

	data["occupied"].iloc[15] = 1
	data["occupied"].iloc[4320] = 1
	data["occupied"].iloc[-15] = 1

	ones_idx = np.argwhere(data["occupied"].values == 1) # check output
	ones_idx = ones_idx.reshape((len(ones_idx),))

	# time_win = quad_win(5, 10, len(data))
	# time_win = V_win(5, 10, len(data))
	time_win = cos_win(5, 10, len(data))

	# print("Length",len(time_win))
	# print("First",time_win[0])
	# print("Last",time_win[-1])
	# print("Min",min(time_win))
	# print("Max",max(time_win))
	plt.plot(time_win)


	for idx in ones_idx:
		# Timewindow ffill
		data["occupied"].iloc[idx:idx+time_win[idx]+1] = 1
		# Timewindow bfill
		data["occupied"].iloc[idx-time_win[idx]:idx] = 1



