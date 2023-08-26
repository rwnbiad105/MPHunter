import sys
import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/DataShare/"


def union(input_size=15, res_API_num=200):
	n = input_size
	top = res_API_num

	list_dic = {}

	for j in range(n):
		read_path = PROJECT_DIR + "/FTQ2Result/ft.Q2." + str(j + 1) + ".rules"
		with open(read_path, "r") as input:
			content = input.read().split('\n')
			i = 0

			for lines in content:
				linelist = lines.split()
				print(linelist)
				if len(linelist) < 2:
					continue
				i = i + 1
				if i > top:
					break
				k = linelist[0]

				#print(linelist[2])
				if not k in list_dic:
					list_dic[k] = linelist[1]

			print(j)

	output = open(PROJECT_DIR + "Union.txt", "a")
	list_dic = sorted(list_dic.items(), key=lambda d: d[1], reverse=True)
	for [list_dic_key, list_dic_value] in list_dic:
		print(list_dic_key + " " + list_dic_value)
		output.write(list_dic_key + " " + list_dic_value + "\n")
	output.close()


if __name__ == "__main__":
	PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../DataShare/"
	input_size = 15
	res_API_num = 200
	union(input_size, res_API_num)
