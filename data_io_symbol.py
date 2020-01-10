import numpy as np
import torch



def load_input(input_file, bit, num_cursor):
	data_input = input_file
	bit=int(bit)
	with open(data_input,"r") as in_data:
		input_lines = in_data.readlines()
	input_1D = []

	in_data.close()
	max_value = 0
	min_value = 64
	for i_line in input_lines:
#		if(int(i_line,2) > max_value):
#			max_value = int(i_line,2)
#		if(int(i_line,2) < min_value):
#			min_value = int(i_line,2)
		input_1D.append(int(i_line,2))  #rstrip : delete spaces on the right side of text
					      #[-1*bit:] : read from position end-6 to end
	input_2D = np.zeros((num_cursor,len(input_1D)-(num_cursor-1)))
	for input_len in range(len(input_1D)-(num_cursor-1)) :
		for num in range(num_cursor) :
			input_2D[num,input_len]=input_1D[input_len+num] #list(text) : split text letter by letter
	input_2D=np.transpose(input_2D)
#	l = input_2D.copy()
#	l[:][:] = min_value
#	input_2D=input_2D-min_value
#	max_value = max_value-min_value
#	print(input_2D)
#	print(max_value)
#	print(min_value)
	input_2D=input_2D/63.0
#	print(input_2D)
	k = input_2D.copy()
	k[:][:] = -32.0/63.0
	input_2D=input_2D+k
#	print(input_2D)
#	print(max_value)
	return input_2D


def load_output(output_file, bit):

	data_output = output_file
	bit=int(bit)
	with open(data_output,"r") as out_data:
		output_lines = out_data.readlines()
	output = []

	out_data.close()	
	for o_line in output_lines:
		output.append(o_line.rstrip()[-1*bit:])  #rstrip : delete spaces on the right side of text
	output_2D = np.zeros((2**bit,len(output)))
	output_1D = np.zeros((len(output)))
	for output_len in range(len(output)) :
#		output_2D[:,output_len]=list(map(int,output_1D[output_len])) #list(text) : split text letter by letter
		output_2D[int(output[output_len],2),output_len]=1 #list(text) : split text letter by letter
		output_1D[output_len] = int(output[output_len],2)
	output_2D = np.transpose(output_2D)
	return output_2D, output_1D

def write_output(output_list, output_file):

	with open(output_file, "w") as out_data:
		for item in output_list :
			out_data.write("%s\n" % item)
		out_data.close()

def write_weight(model, output_file) :

	with open(output_file, "w") as out_weight:
		for param in model.parameters() :
			out_weight.write("%s\n" % param.data)
		out_weight.close()
